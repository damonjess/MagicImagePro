//
// inpaint_engine.h - content-aware fill core, shared by the Android JNI
// engine and the desktop test harness.
//
// The diffusion-based OpenCV inpainters (Telea/NS) smear rim colors across
// large holes, which is why removed objects left an obvious smudge. This
// engine adds a PatchMatch-style patch fill: it searches the rest of the
// photo for real texture patches and copies them into the hole, then
// matches low-frequency color and blends the seam.
//
// Expected color format is 8-bit BGR (OpenCV convention).
//
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

namespace inpaint_engine {// ---------------------------------------------------------------------------
// PatchMatch: fill the hole by copying coherent patches of REAL texture from
// the rest of the photo, refining the copy offsets across a small pyramid.
//
// Coarse levels first (global structure: pavement stays pavement, grass stays
// grass), then full resolution (sharp texture). Rendering blends a hard copy
// from the pixel's own best patch (keeps texture crisp) with the neighbor
// vote average (hides seams between patch regions).
// ---------------------------------------------------------------------------

class PatchMatchFill {
public:
    PatchMatchFill(const cv::Mat& src, const cv::Mat& holeMask, uint64_t seed = 12345)
        : rng_(seed) {
        buildPyramid(src, holeMask);
    }

    cv::Mat run() {
        // Coarse-to-fine over the pyramid.
        for (int lvl = (int)levels_.size() - 1; lvl >= 0; --lvl) {
            Level& L = levels_[lvl];
            if (!L.hasHole) continue;
            if (lvl == (int)levels_.size() - 1) {
                initOffsets(L);
            } else {
                upsampleOffsets(levels_[lvl + 1], L);
            }
            const int iters = (lvl == 0) ? 4 : 3;
            for (int it = 0; it < iters; ++it) {
                scanline(L, it);
                aggregate(L);
            }
        }
        return render();
    }

private:
    struct Level {
        cv::Mat img;              // working image (hole estimate evolves here)
        cv::Mat hole;             // 255 = hole
        std::vector<bool> isHole, isSource, hasOff;
        std::vector<int> sourceIdx;
        std::vector<int> offX, offY;
        cv::Mat filled;           // latest vote render (BGR)
        int w = 0, h = 0, n = 0;
        bool hasHole = false;
    };

    static inline float lum(const uchar* bgr, int i) {
        return 0.114f * bgr[i * 3] + 0.587f * bgr[i * 3 + 1] + 0.299f * bgr[i * 3 + 2];
    }

    void buildPyramid(const cv::Mat& src, const cv::Mat& holeMask) {
        cv::Mat img = src.clone();
        cv::Mat hole;
        cv::threshold(holeMask, hole, 127, 255, cv::THRESH_BINARY);
        while (true) {
            Level L;
            L.img = img.clone();
            L.hole = hole.clone();
            L.w = img.cols; L.h = img.rows; L.n = L.w * L.h;
            L.isHole.assign(L.n, false);
            L.isSource.assign(L.n, false);
            const uchar* hp = L.hole.ptr<uchar>(0);
            for (int i = 0; i < L.n; ++i) L.isHole[i] = hp[i] > 0;
            // Source pixels: outside the hole and 4px away from it (rim may
            // still hold traces of the removed object).
            cv::Mat rimFree;
            cv::erode(~L.hole, rimFree,
                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
            const uchar* sp = rimFree.ptr<uchar>(0);
            for (int i = 0; i < L.n; ++i) L.isSource[i] = sp[i] > 0;
            for (int i = 0; i < L.n; ++i) if (L.isSource[i]) L.sourceIdx.push_back(i);
            L.offX.assign(L.n, 0); L.offY.assign(L.n, 0); L.hasOff.assign(L.n, false);
            L.hasHole = cv::countNonZero(L.hole) > 0;
            levels_.push_back(L);
            if (std::max(img.cols, img.rows) <= 128 || !levels_.back().hasHole) break;
            cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cv::resize(hole, hole, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
            cv::threshold(hole, hole, 0, 255, cv::THRESH_BINARY);
        }
    }

    void initOffsets(Level& L) {
        for (int i = 0; i < L.n; ++i) {
            if (!L.isHole[i]) continue;
            int x = i % L.w, y = i / L.w;
            const int dxs[4] = {-1, 1, 0, 0};
            const int dys[4] = {0, 0, -1, 1};
            bool placed = false;
            for (int k = 0; k < 4 && !placed; ++k) {
                int x2 = x + dxs[k], y2 = y + dys[k];
                while (x2 >= 0 && x2 < L.w && y2 >= 0 && y2 < L.h && L.isHole[y2 * L.w + x2]) {
                    x2 += dxs[k]; y2 += dys[k];
                }
                if (x2 >= 0 && x2 < L.w && y2 >= 0 && y2 < L.h && L.isSource[y2 * L.w + x2]) {
                    L.offX[i] = x2 - x; L.offY[i] = y2 - y; L.hasOff[i] = true; placed = true;
                }
            }
            if (!placed && !L.sourceIdx.empty()) {
                int s = L.sourceIdx[rng_() % L.sourceIdx.size()];
                L.offX[i] = s % L.w - x; L.offY[i] = s / L.w - y; L.hasOff[i] = true;
            }
        }
    }

    void upsampleOffsets(const Level& fine_from, Level& to) {
        // 'from' is the coarser level; scale its offsets x2 into 'to'.
        for (int i = 0; i < to.n; ++i) {
            if (!to.isHole[i]) continue;
            int x = i % to.w, y = i / to.w;
            int fx = std::min(fine_from.w - 1, x / 2);
            int fy = std::min(fine_from.h - 1, y / 2);
            int j = fy * fine_from.w + fx;
            if (!fine_from.hasOff[j]) continue;
            int sx = x * 2 + fine_from.offX[j] * 2;
            int sy = y * 2 + fine_from.offY[j] * 2;
            sx = std::clamp(sx, 0, to.w - 1);
            sy = std::clamp(sy, 0, to.h - 1);
            if (to.isSource[sy * to.w + sx]) {
                to.offX[i] = sx - x; to.offY[i] = sy - y; to.hasOff[i] = true;
            }
        }
        // Any hole pixel still without an offset: nearest-known init.
        for (int i = 0; i < to.n; ++i) {
            if (!to.isHole[i] || to.hasOff[i]) continue;
            int x = i % to.w, y = i / to.w;
            const int dxs[4] = {-1, 1, 0, 0};
            const int dys[4] = {0, 0, -1, 1};
            bool placed = false;
            for (int k = 0; k < 4 && !placed; ++k) {
                int x2 = x + dxs[k], y2 = y + dys[k];
                while (x2 >= 0 && x2 < to.w && y2 >= 0 && y2 < to.h && to.isHole[y2 * to.w + x2]) {
                    x2 += dxs[k]; y2 += dys[k];
                }
                if (x2 >= 0 && x2 < to.w && y2 >= 0 && y2 < to.h && to.isSource[y2 * to.w + x2]) {
                    to.offX[i] = x2 - x; to.offY[i] = y2 - y; to.hasOff[i] = true; placed = true;
                }
            }
            if (!placed && !to.sourceIdx.empty()) {
                int s = to.sourceIdx[rng_() % to.sourceIdx.size()];
                to.offX[i] = s % to.w - x; to.offY[i] = s / to.w - y; to.hasOff[i] = true;
            }
        }
    }

    // SSD between the 7x7 neighborhood around hole pixel (x,y) mapped through
    // offset (dx,dy) and the neighborhood around the source. Compares against
    // the level's working image (known pixels are original, hole pixels hold
    // the current textured estimate).
    float patchCost(const Level& L, int x, int y, int dx, int dy) const {
        float cost = 0.f;
        int cnt = 0;
        for (int ny = y - 3; ny <= y + 3; ++ny) {
            for (int nx = x - 3; nx <= x + 3; ++nx) {
                if (nx < 0 || nx >= L.w || ny < 0 || ny >= L.h) continue;
                int a = ny * L.w + nx;
                int bx = nx + dx, by = ny + dy;
                if (bx < 0 || bx >= L.w || by < 0 || by >= L.h) return 1e9f;
                int b = by * L.w + bx;
                if (L.isHole[b]) continue;  // sources must be known texture
                const uchar* pa = L.img.data + a * 3;
                const uchar* pb = L.img.data + b * 3;
                float dr = pa[0] - pb[0], dg = pa[1] - pb[1], db = pa[2] - pb[2];
                cost += dr * dr + dg * dg + db * db;
                ++cnt;
            }
        }
        return cnt > 0 ? cost / cnt : 1e9f;
    }

    void scanline(Level& L, int iter) {
        const bool reverse = (iter & 1) != 0;
        for (int yy = 0; yy < L.h; ++yy) {
            const int y = reverse ? L.h - 1 - yy : yy;
            for (int xx = 0; xx < L.w; ++xx) {
                const int x = reverse ? L.w - 1 - xx : xx;
                const int i = y * L.w + x;
                if (!L.isHole[i] || !L.hasOff[i]) continue;

                const int pxs[2] = {reverse ? 1 : -1, 0};
                const int pys[2] = {0, reverse ? 1 : -1};
                float best = patchCost(L, x, y, L.offX[i], L.offY[i]);
                for (int k = 0; k < 2; ++k) {
                    int nx = x + pxs[k], ny = y + pys[k];
                    if (nx < 0 || nx >= L.w || ny < 0 || ny >= L.h) continue;
                    int j = ny * L.w + nx;
                    if (!L.hasOff[j]) continue;
                    float c = patchCost(L, x, y, L.offX[j], L.offY[j]);
                    if (c < best) { best = c; L.offX[i] = L.offX[j]; L.offY[i] = L.offY[j]; }
                }

                int wx = L.w / 2, wy = L.h / 2;
                const int cx = x + L.offX[i], cy = y + L.offY[i];
                while (wx > 0 || wy > 0) {
                    int sx = cx + (int)((rng_() % (2 * wx + 1)) - wx);
                    int sy = cy + (int)((rng_() % (2 * wy + 1)) - wy);
                    if (sx >= 0 && sx < L.w && sy >= 0 && sy < L.h && L.isSource[sy * L.w + sx]) {
                        float c = patchCost(L, x, y, sx - x, sy - y);
                        if (c < best) { best = c; L.offX[i] = sx - x; L.offY[i] = sy - y; }
                    }
                    wx /= 2; wy /= 2;
                }
            }
        }
    }

    // Vote: average of neighbor patches' sources, blended with the hard copy
    // from the pixel's own patch. The blend is ADAPTIVE: where the pixel's
    // neighbors disagree about source offsets (region seams), vote-weight is
    // raised so the transition becomes a gradual organic mix; where they
    // agree, the hard copy dominates and texture stays sharp.
    void aggregate(Level& L) {
        L.filled = L.img.clone();
        for (int y = 0; y < L.h; ++y) {
            for (int x = 0; x < L.w; ++x) {
                int i = y * L.w + x;
                if (!L.isHole[i] || !L.hasOff[i]) continue;
                int ownX = x + L.offX[i], ownY = y + L.offY[i];
                if (ownX < 0 || ownX >= L.w || ownY < 0 || ownY >= L.h) continue;
                const uchar* own = L.img.data + (ownY * L.w + ownX) * 3;

                // Measure local offset disagreement (seam detector).
                float disp = 0.f;
                int dispN = 0;
                for (int qy = y - 2; qy <= y + 2; ++qy) {
                    for (int qx = x - 2; qx <= x + 2; ++qx) {
                        if (qx < 0 || qx >= L.w || qy < 0 || qy >= L.h) continue;
                        int j = qy * L.w + qx;
                        if (!L.isHole[j] || !L.hasOff[j]) continue;
                        float odx = float(L.offX[j] - L.offX[i]);
                        float ody = float(L.offY[j] - L.offY[i]);
                        disp += std::sqrt(odx * odx + ody * ody);
                        ++dispN;
                    }
                }
                float seam = dispN > 0 ? std::clamp(disp / dispN / 24.0f, 0.f, 1.f) : 0.f;
                float voteShare = 0.35f + 0.5f * seam; // 0.35 coherent .. 0.85 seam

                float sr = 0, sg = 0, sb = 0, wt = 0;
                for (int qy = y - 2; qy <= y + 2; ++qy) {
                    for (int qx = x - 2; qx <= x + 2; ++qx) {
                        if (qx < 0 || qx >= L.w || qy < 0 || qy >= L.h) continue;
                        int j = qy * L.w + qx;
                        if (!L.isHole[j] || !L.hasOff[j]) continue;
                        int sx = qx + L.offX[j], sy = qy + L.offY[j];
                        if (sx < 0 || sx >= L.w || sy < 0 || sy >= L.h) continue;
                        if (L.isHole[sy * L.w + sx]) continue;
                        const uchar* ps = L.img.data + (sy * L.w + sx) * 3;
                        float odx = float(L.offX[j] - L.offX[i]);
                        float ody = float(L.offY[j] - L.offY[i]);
                        float coh = 1.f / (1.f + (odx * odx + ody * ody) / 256.f);
                        float dist2 = float((qx - x) * (qx - x) + (qy - y) * (qy - y));
                        float sp = 1.f / (1.f + dist2 / 8.f);
                        float wgt = (0.25f + 0.75f * coh) * sp;
                        sb += ps[0] * wgt; sg += ps[1] * wgt; sr += ps[2] * wgt;
                        wt += wgt;
                    }
                }
                cv::Vec3b& out = L.filled.at<cv::Vec3b>(y, x);
                if (wt > 0.f) {
                    float a = voteShare;
                    out = cv::Vec3b(
                        cv::saturate_cast<uchar>((1.f - a) * own[0] + a * sb / wt),
                        cv::saturate_cast<uchar>((1.f - a) * own[1] + a * sg / wt),
                        cv::saturate_cast<uchar>((1.f - a) * own[2] + a * sr / wt));
                } else {
                    out = cv::Vec3b(own[0], own[1], own[2]);
                }
            }
        }
        // Evolve the working estimate for the next iteration / finer level.
        for (int i = 0; i < L.n; ++i) {
            if (L.isHole[i] && L.hasOff[i]) {
                L.img.data[i * 3]     = L.filled.data[i * 3];
                L.img.data[i * 3 + 1] = L.filled.data[i * 3 + 1];
                L.img.data[i * 3 + 2] = L.filled.data[i * 3 + 2];
            }
        }
    }

    cv::Mat render() {
        Level& L = levels_[0];
        cv::Mat out = L.img.clone();
        if (!L.filled.empty()) {
            for (int i = 0; i < L.n; ++i) {
                if (L.isHole[i]) {
                    out.data[i * 3]     = L.filled.data[i * 3];
                    out.data[i * 3 + 1] = L.filled.data[i * 3 + 1];
                    out.data[i * 3 + 2] = L.filled.data[i * 3 + 2];
                }
            }
        }
        return out;
    }

    std::vector<Level> levels_;
    cv::RNG rng_;
};

// ---------------------------------------------------------------------------
// Low-frequency color match: aligns the fill's smooth component with the
// original photo around the hole so brightness/tint transitions are seamless.
// ---------------------------------------------------------------------------

static void matchLowFrequency(cv::Mat& filled, const cv::Mat& original, const cv::Mat& hole) {
    const int k = std::max(21, std::min(original.cols, original.rows) / 30) | 1;

    cv::Mat fBlur, oBlur;
    cv::GaussianBlur(filled, fBlur, cv::Size(k, k), 0);
    cv::GaussianBlur(original, oBlur, cv::Size(k, k), 0);

    // CRITICAL: the original's low frequency inside the hole still contains
    // the removed object's colors. Discard it and extend the SURROUNDINGS'
    // low frequency into the hole instead, so the correction is spatially
    // adaptive (pavement area adopts pavement tones, grass area grass tones)
    // and cannot pull the fill back toward the object's own color.
    cv::Mat oBlur8;
    oBlur.convertTo(oBlur8, CV_8UC3);
    // Low-frequency content only: run the extension at quarter res.
    cv::Mat oSmall, holeSmall, oExtendSmall;
    cv::resize(oBlur8, oSmall, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::resize(hole, holeSmall, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
    cv::threshold(holeSmall, holeSmall, 0, 255, cv::THRESH_BINARY);
    cv::inpaint(oSmall, holeSmall, oExtendSmall, 6, cv::INPAINT_TELEA);
    cv::Mat oExtend;
    cv::resize(oExtendSmall, oExtend, original.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat oExtendF;
    oExtend.convertTo(oExtendF, CV_32FC3);

    cv::Mat fBlurF;
    fBlur.convertTo(fBlurF, CV_32FC3);

    // Feather the correction from the rim (no change) into the interior.
    cv::Mat dist;
    cv::distanceTransform(hole, dist, cv::DIST_L2, 3);
    const float ramp = 12.0f;

    cv::Mat filledF;
    filled.convertTo(filledF, CV_32FC3);
    for (int y = 0; y < filled.rows; ++y) {
        const uchar* m = hole.ptr<uchar>(y);
        const float* d = dist.ptr<float>(y);
        cv::Vec3f* fp = filledF.ptr<cv::Vec3f>(y);
        const cv::Vec3f* fb = fBlurF.ptr<cv::Vec3f>(y);
        const cv::Vec3f* ob = oExtendF.ptr<cv::Vec3f>(y);
        for (int x = 0; x < filled.cols; ++x) {
            if (m[x] == 0) continue;
            float t = std::clamp(d[x] / ramp, 0.0f, 1.0f);
            t = t * t * (3.0f - 2.0f * t);
            if (t <= 0.f) continue;
            cv::Vec3f out = fp[x];
            for (int c = 0; c < 3; ++c) {
                float denom = std::max(fb[x][c], 1.0f);
                float gain = std::clamp(ob[x][c] / denom, 0.7f, 1.4f);
                float corrected = fp[x][c] * gain;
                out[c] = fp[x][c] * (1.f - t) + corrected * t;
            }
            fp[x] = out;
        }
    }
    filledF.convertTo(filled, CV_8UC3);
}

// ---------------------------------------------------------------------------
// Grain matching: re-inject noise with the same strength as the photo's own
// sensor grain near the hole. AI/diffusion/patch fills all come out slightly
// "too clean", which is one of the main tells of a removed object.
// ---------------------------------------------------------------------------

static float estimateGrain(const cv::Mat& original, const cv::Mat& hole) {
    cv::Mat gray, blur;
    cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
    cv::Mat hf;
    cv::absdiff(gray, blur, hf);

    cv::Mat ring;
    cv::dilate(hole, ring, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(81, 81)));
    cv::erode(ring, ring, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21)));
    cv::subtract(ring, hole, ring);   // exclude rim (may contain object remnants)

    cv::Scalar m, s;
    cv::meanStdDev(hf, m, s, ring);
    return (float)std::min(6.0, std::max(0.0, s[0]));
}

static void applyGrain(cv::Mat& img, const cv::Mat& hole, float std, uint64_t seed = 999) {
    if (std <= 0.3f) return;
    cv::RNG rng(seed);
    const float amp = std * 1.732f * 0.85f; // uniform[-a,a] has std a/sqrt(3)
    for (int y = 0; y < img.rows; ++y) {
        const uchar* m = hole.ptr<uchar>(y);
        cv::Vec3b* p = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (m[x] == 0) continue;
            float nz = (rng.uniform(-1.f, 1.f)) * amp;
            p[x] = cv::Vec3b(
                cv::saturate_cast<uchar>(p[x][0] + nz),
                cv::saturate_cast<uchar>(p[x][1] + nz),
                cv::saturate_cast<uchar>(p[x][2] + nz));
        }
    }
}

// ---------------------------------------------------------------------------
// Seam blend: keep original pixels at the hole rim, fade to the fill a few
// pixels inside. Rim pixels are the ones that most plausibly still contain
// traces of the removed object, so never trust them into the fill.
// ---------------------------------------------------------------------------

static void blendSeam(const cv::Mat& original, cv::Mat& filled, const cv::Mat& hole) {
    cv::Mat dist;
    cv::distanceTransform(hole, dist, cv::DIST_L2, 3);
    const float bandF = 6.0f;

    for (int y = 0; y < filled.rows; ++y) {
        const uchar* m = hole.ptr<uchar>(y);
        const float* d = dist.ptr<float>(y);
        const cv::Vec3b* s = original.ptr<cv::Vec3b>(y);
        cv::Vec3b* f = filled.ptr<cv::Vec3b>(y);
        for (int x = 0; x < filled.cols; ++x) {
            if (m[x] == 0) continue;
            float t = std::clamp(d[x] / bandF, 0.0f, 1.0f);
            t = t * t * (3.0f - 2.0f * t); // smoothstep
            const cv::Vec3b& a = s[x];
            cv::Vec3b& b = f[x];
            b = cv::Vec3b(
                cv::saturate_cast<uchar>(a[0] + (b[0] - a[0]) * t),
                cv::saturate_cast<uchar>(a[1] + (b[1] - a[1]) * t),
                cv::saturate_cast<uchar>(a[2] + (b[2] - a[2]) * t));
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

static cv::Mat fillHole(const cv::Mat& srcBgr, const cv::Mat& holeMask) {
    CV_Assert(srcBgr.type() == CV_8UC3);
    CV_Assert(holeMask.type() == CV_8UC1);
    CV_Assert(srcBgr.size() == holeMask.size());

    cv::Mat hole;
    cv::threshold(holeMask, hole, 127, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(hole) == 0) return srcBgr.clone();

    // 1) Initial diffusion fill (gives PatchMatch reasonable starting colors).
    cv::Mat init;
    cv::inpaint(srcBgr, hole, init, 7.0, cv::INPAINT_TELEA);

    // 2) Multi-scale PatchMatch: replace the smeared interior with real
    //    copied texture, coarse structure first, sharp texture last.
    PatchMatchFill pm(init, hole);
    cv::Mat filled = pm.run();

    // 3) Low-frequency color match against the original surroundings.
    matchLowFrequency(filled, srcBgr, hole);

    // 4) Tight seam blend at the rim.
    blendSeam(srcBgr, filled, hole);

    // 5) Grain matching.
    float grain = estimateGrain(srcBgr, hole);
    applyGrain(filled, hole, grain);

    return filled;
}

// ---------------------------------------------------------------------------
// Texture transfer: borrow the photo's own fine detail for a generated fill.
//
// A generative fill gets the global structure right - where the pavement ends
// and the grass begins - but it runs at a small fixed resolution, so a large
// hole comes back visibly smooth. Real grass, concrete and fabric are full of
// fine detail that a smooth fill does not have, and that missing detail is the
// strongest remaining tell that something was removed.
//
// Every masked pixel looks for the unmasked pixel whose blurred surroundings
// are most similar to its own, and copies that pixel's high-frequency detail on
// top of the fill. The structure comes from the model, the texture from the
// photo itself, so the hole ends up with the same grain and structure as its
// surroundings instead of a soft patch.
//
// The detail band is deliberately narrow (detailSigma): only the photo's fine
// grain is borrowed, never its structure. Copying a wider band drags real
// shapes along with the offset field and shows up as swirls in the fill, which
// was measurably worse than leaving the hole smooth.
// ---------------------------------------------------------------------------

static void transferTexture(cv::Mat& img, const cv::Mat& original,
                            const cv::Mat& hole, float strength = 0.8f,
                            int cell = 14, double offsetSigma = 0.5,
                            double detailSigma = 2.5, double postSigma = 12.0) {
    CV_Assert(img.size() == original.size());
    const int W = img.cols, H = img.rows;
    if (cv::countNonZero(hole) == 0) return;

    cv::Mat origF, imgF;
    original.convertTo(origF, CV_32FC3);
    img.convertTo(imgF, CV_32FC3);

    cv::Mat baseOrig, baseFill;
    cv::GaussianBlur(origF, baseOrig, cv::Size(0, 0), detailSigma);
    cv::GaussianBlur(imgF, baseFill, cv::Size(0, 0), detailSigma);
    cv::Mat detail = origF - baseOrig;   // real fine detail outside the hole

    // --- coarse nearest-neighbour match on blurred colours -----------------
    const int gw = (W + cell - 1) / cell;
    const int gh = (H + cell - 1) / cell;
    cv::Mat holeF, cov, coarse;
    hole.convertTo(holeF, CV_32F, 1.0 / 255.0);
    cv::resize(holeF, cov, cv::Size(gw, gh), 0, 0, cv::INTER_AREA);
    cv::resize(baseFill, coarse, cv::Size(gw, gh), 0, 0, cv::INTER_AREA);

    std::vector<cv::Point> holeCells, sourceCells;
    std::vector<cv::Vec3f> holeVal, sourceVal;
    for (int gy = 0; gy < gh; ++gy) {
        const float* cp = cov.ptr<float>(gy);
        const cv::Vec3f* fp = coarse.ptr<cv::Vec3f>(gy);
        for (int gx = 0; gx < gw; ++gx) {
            if (cp[gx] > 0.002f) {
                holeCells.emplace_back(gx, gy);
                holeVal.push_back(fp[gx]);
            } else if (cp[gx] <= 0.0f) {
                // Donor cells must be clear of the hole, including their
                // neighbours, so no removed-object remnant leaks into the fill.
                bool clear = true;
                for (int ny = -1; ny <= 1 && clear; ++ny) {
                    for (int nx = -1; nx <= 1; ++nx) {
                        const int cx = gx + nx, cy = gy + ny;
                        if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
                        if (cov.at<float>(cy, cx) > 0.0f) { clear = false; break; }
                    }
                }
                if (clear) {
                    sourceCells.emplace_back(gx, gy);
                    sourceVal.push_back(fp[gx]);
                }
            }
        }
    }
    if (holeCells.empty() || sourceCells.empty()) return;

    cv::Mat offX = cv::Mat::zeros(gh, gw, CV_32F);
    cv::Mat offY = cv::Mat::zeros(gh, gw, CV_32F);
    for (size_t i = 0; i < holeCells.size(); ++i) {
        const cv::Vec3f& q = holeVal[i];
        float best = FLT_MAX;
        int bestJ = -1;
        for (size_t j = 0; j < sourceCells.size(); ++j) {
            const cv::Vec3f& s = sourceVal[j];
            const float dr = q[0] - s[0], dg = q[1] - s[1], db = q[2] - s[2];
            const float d = dr * dr + dg * dg + db * db;
            if (d < best) { best = d; bestJ = (int)j; }
        }
        if (bestJ < 0) continue;
        const cv::Point& h = holeCells[i];
        const cv::Point& s = sourceCells[bestJ];
        offX.at<float>(h) = static_cast<float>((s.x - h.x) * cell);
        offY.at<float>(h) = static_cast<float>((s.y - h.y) * cell);
    }

    // Barely smooth the donor field. Smoothing it more makes neighbouring
    // pixels pull their detail from places that rotate and scale relative to
    // each other, and warping fine texture that way draws faint spirals across
    // the result. A lightly smoothed field keeps the texture coherent without
    // deforming it; measured against the photo's own texture this needed a
    // lower strength too, so both knobs moved together.
    // NB: this field lives on the coarse grid, so sigma is measured in cells.
    // A tiny sigma means "keep each cell's own donor", which rigidly copies
    // blocks of texture instead of warping it - see the note in the apply loop.
    if (offsetSigma > 0.05) {
        cv::GaussianBlur(offX, offX, cv::Size(0, 0), offsetSigma);
        cv::GaussianBlur(offY, offY, cv::Size(0, 0), offsetSigma);
    }
    cv::Mat offXBig, offYBig;
    cv::resize(offX, offXBig, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    cv::resize(offY, offYBig, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

    // Feather the added detail in from the rim so the hole boundary stays exact.
    cv::Mat dist;
    cv::distanceTransform(hole, dist, cv::DIST_L2, 3);
    const float ramp = std::max(16.0f, static_cast<float>(cell) * 2.5f);

    // Sample the borrowed detail through the donor field first. Warping fine
    // texture with a smooth field inevitably folds it into low-frequency
    // whorls, and those whorls are far more visible than the grain they came
    // from - they read as faint spirals drawn on the pavement. Removing the
    // warped sample's own local average leaves only the texture that was
    // wanted, so the field can be as smooth as the match needs.
    cv::Mat warped;
    detail.copyTo(warped);
    cv::Mat sampled = cv::Mat::zeros(H, W, CV_8U);

    for (int y = 0; y < H; ++y) {
        const uchar* m = hole.ptr<uchar>(y);
        const float* ox = offXBig.ptr<float>(y);
        const float* oy = offYBig.ptr<float>(y);
        uchar* sp = sampled.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            if (m[x] == 0) continue;
            // Walk back towards the pixel if the donor drifted into the hole.
            float scale = 1.0f;
            int dx = 0, dy = 0;
            bool ok = false;
            for (int attempt = 0; attempt < 6 && !ok; ++attempt) {
                dx = std::clamp(x + static_cast<int>(std::lround(ox[x] * scale)), 0, W - 1);
                dy = std::clamp(y + static_cast<int>(std::lround(oy[x] * scale)), 0, H - 1);
                ok = hole.at<uchar>(dy, dx) == 0;
                scale *= 0.5f;
            }
            if (!ok) {
                warped.at<cv::Vec3f>(y, x) = cv::Vec3f(0.f, 0.f, 0.f);
                continue;
            }
            warped.at<cv::Vec3f>(y, x) = detail.at<cv::Vec3f>(dy, dx);
            sp[x] = 1;
        }
    }

    if (postSigma > 0.5) {
        cv::Mat warpedLow;
        cv::GaussianBlur(warped, warpedLow, cv::Size(0, 0), postSigma);
        cv::subtract(warped, warpedLow, warped);
    }

    for (int y = 0; y < H; ++y) {
        const uchar* m = hole.ptr<uchar>(y);
        const uchar* sp = sampled.ptr<uchar>(y);
        const float* d = dist.ptr<float>(y);
        const cv::Vec3f* nzp = warped.ptr<cv::Vec3f>(y);
        cv::Vec3f* p = imgF.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            if (m[x] == 0 || sp[x] == 0) continue;
            float t = std::clamp(d[x] / ramp, 0.0f, 1.0f);
            t = t * t * (3.0f - 2.0f * t);
            if (t <= 0.f) continue;
            const float k = strength * t;
            p[x][0] = std::clamp(p[x][0] + nzp[x][0] * k, 0.0f, 255.0f);
            p[x][1] = std::clamp(p[x][1] + nzp[x][1] * k, 0.0f, 255.0f);
            p[x][2] = std::clamp(p[x][2] + nzp[x][2] * k, 0.0f, 255.0f);
        }
    }
    imgF.convertTo(img, CV_8UC3);
}

// ---------------------------------------------------------------------------
// Refine an externally produced fill (e.g. a generative model's output).
//
// The model supplies the structure; transferTexture then borrows the photo's
// own fine detail on top of it, and the seam is blended tight against the
// original. Grain is left to the caller so it is only applied once.
// ---------------------------------------------------------------------------

static cv::Mat refineFill(const cv::Mat& srcBgr, const cv::Mat& holeMask,
                          const cv::Mat& initBgr) {
    CV_Assert(srcBgr.type() == CV_8UC3);
    CV_Assert(holeMask.type() == CV_8UC1);
    CV_Assert(srcBgr.size() == holeMask.size());
    CV_Assert(initBgr.size() == srcBgr.size());

    cv::Mat hole;
    cv::threshold(holeMask, hole, 127, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(hole) == 0) return srcBgr.clone();

    // Inside the hole take the generated fill; outside it take the real photo.
    cv::Mat filled = initBgr.clone();
    srcBgr.copyTo(filled, ~hole);

    transferTexture(filled, srcBgr, hole);
    blendSeam(srcBgr, filled, hole);

    return filled;
}

} // namespace inpaint_engine
