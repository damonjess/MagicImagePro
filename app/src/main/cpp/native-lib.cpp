#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include "inpaint_engine.h"
#include <algorithm>
#include <vector>

#define LOG_TAG "NativeProcessor"
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

// ----------------------------------------------------------------------------
// Seam composite
//
// Blends the inpainted result into the original with a tight edge-fade driven
// by the mask's distance transform: at the hole rim we keep the original
// pixels (perfect color/texture continuity), fading to the pure inpainted
// content a few pixels deeper inside. Nothing outside the mask is ever
// modified, and no global color statistics are applied.
//
// The previous pipeline (border color-gain + cv::seamlessClone Poisson
// diffusion + a wide dilated feather) repeatedly smeared flat color over the
// hole and bled surrounding colors into it, which is what produced the
// blurry "the dog is still faintly there" look even when LaMa's output was
// good. With a real generative fill underneath, the seam only needs a few
// pixels of blending.
// ----------------------------------------------------------------------------
static cv::Mat seamlessCompositePipeline(const cv::Mat& srcRgb,
                                          const cv::Mat& inRgb,
                                          const cv::Mat& binMask) {
    if (cv::countNonZero(binMask) == 0) {
        return srcRgb.clone();
    }

    const int maxDim = std::max(srcRgb.cols, srcRgb.rows);
    const int band = std::clamp(maxDim / 400, 3, 8);
    const float bandF = static_cast<float>(band);

    cv::Mat distMap;
    cv::distanceTransform(binMask, distMap, cv::DIST_L2, 3);

    cv::Mat out = srcRgb.clone();
    for (int y = 0; y < srcRgb.rows; ++y) {
        const uchar* mPtr = binMask.ptr<uchar>(y);
        const float* dPtr = distMap.ptr<float>(y);
        const cv::Vec3b* sPtr = srcRgb.ptr<cv::Vec3b>(y);
        const cv::Vec3b* iPtr = inRgb.ptr<cv::Vec3b>(y);
        cv::Vec3b* oPtr = out.ptr<cv::Vec3b>(y);

        for (int x = 0; x < srcRgb.cols; ++x) {
            if (mPtr[x] == 0) continue;
            float t = std::clamp(dPtr[x] / bandF, 0.0f, 1.0f);
            t = t * t * (3.0f - 2.0f * t); // smoothstep
            const cv::Vec3b& s = sPtr[x];
            const cv::Vec3b& p = iPtr[x];
            oPtr[x] = cv::Vec3b(
                cv::saturate_cast<uchar>(s[0] + (p[0] - s[0]) * t),
                cv::saturate_cast<uchar>(s[1] + (p[1] - s[1]) * t),
                cv::saturate_cast<uchar>(s[2] + (p[2] - s[2]) * t));
        }
    }
    return out;
}

static inline cv::Mat inpaintHybrid(const cv::Mat& srcRgb, const cv::Mat& maskFull,
                                     int radius) {
    // Fine passes at full resolution keep edges and structure near the rim.
    cv::Mat outTelea, outNs;
    cv::inpaint(srcRgb, maskFull, outTelea, static_cast<double>(radius), cv::INPAINT_TELEA);
    cv::inpaint(srcRgb, maskFull, outNs, static_cast<double>(radius + 2), cv::INPAINT_NS);

    cv::Mat edgeMask, sobelX, sobelY;
    cv::Sobel(srcRgb, sobelX, CV_32F, 1, 0, 3);
    cv::Sobel(srcRgb, sobelY, CV_32F, 0, 1, 3);
    cv::magnitude(sobelX, sobelY, edgeMask);
    edgeMask = edgeMask / 255.0;

    // fineMix = (1-w)*Telea + w*NS, weighted by local edge strength.
    std::vector<cv::Mat> teleaCh, nsCh;
    cv::split(outTelea, teleaCh);
    cv::split(outNs, nsCh);
    cv::Mat fineMix = cv::Mat::zeros(srcRgb.size(), CV_32FC3);
    for (int y = 0; y < srcRgb.rows; ++y) {
        const uchar* mPtr = maskFull.ptr<uchar>(y);
        cv::Vec3f* dst = fineMix.ptr<cv::Vec3f>(y);
        for (int x = 0; x < srcRgb.cols; ++x) {
            if (mPtr[x] == 0) continue;
            float e = (edgeMask.at<cv::Vec3f>(y, x)[0] +
                       edgeMask.at<cv::Vec3f>(y, x)[1] +
                       edgeMask.at<cv::Vec3f>(y, x)[2]) / 3.0f;
            float w = (e < 0.05f) ? 0.3f : (e > 0.2f ? 0.7f : 0.5f);
            for (int c = 0; c < 3; ++c) {
                float t = teleaCh[c].at<float>(y, x);
                float n = nsCh[c].at<float>(y, x);
                dst[x][c] = t * (1.0f - w) + n * w;
            }
        }
    }

    // Coarse pass at 1/4 resolution: diffusion inpainting propagates structure
    // much better when run small, giving smoother low-frequency content across
    // large holes instead of the telltale bright smear.
    cv::Mat small, smallMask, smallFilled, up8, coarse;
    cv::resize(srcRgb, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::resize(maskFull, smallMask, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
    cv::threshold(smallMask, smallMask, 0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(smallMask) > 0) {
        cv::inpaint(small, smallMask, smallFilled, 8.0, cv::INPAINT_TELEA);
        cv::resize(smallFilled, up8, srcRgb.size(), 0, 0, cv::INTER_CUBIC);
        up8.convertTo(coarse, CV_32FC3);
    } else {
        srcRgb.convertTo(coarse, CV_32FC3);
    }

    // Blend fine -> coarse by distance from the hole edge: the rim keeps the
    // structure-preserving fine result, the deep interior gets the smoother
    // coarse fill.
    cv::Mat distMap;
    cv::distanceTransform(maskFull, distMap, cv::DIST_L2, 3);
    const float deep = 40.0f;
    cv::Mat combined;
    fineMix.convertTo(combined, CV_8UC3);
    for (int y = 0; y < srcRgb.rows; ++y) {
        const uchar* mPtr = maskFull.ptr<uchar>(y);
        const float* dPtr = distMap.ptr<float>(y);
        const cv::Vec3f* cPtr = coarse.ptr<cv::Vec3f>(y);
        cv::Vec3b* oPtr = combined.ptr<cv::Vec3b>(y);
        for (int x = 0; x < srcRgb.cols; ++x) {
            if (mPtr[x] == 0) continue;
            float cw = std::clamp(dPtr[x] / deep, 0.0f, 1.0f);
            cv::Vec3f f = fineMix.at<cv::Vec3f>(y, x);
            oPtr[x] = cv::Vec3b(
                cv::saturate_cast<uchar>(f[0] * (1.0f - cw) + cPtr[x][0] * cw),
                cv::saturate_cast<uchar>(f[1] * (1.0f - cw) + cPtr[x][1] * cw),
                cv::saturate_cast<uchar>(f[2] * (1.0f - cw) + cPtr[x][2] * cw));
        }
    }

    return seamlessCompositePipeline(srcRgb, combined, maskFull);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_magicimagepro_ml_NativeProcessor_processImage(
        JNIEnv* env,
        jobject /* this */,
        jobject original,
        jobject mask,
        jobject outBitmap) {

    AndroidBitmapInfo infoOrig, infoMask, infoOut;
    void *pixelsOrig = nullptr, *pixelsMask = nullptr, *pixelsOut = nullptr;

    if (AndroidBitmap_getInfo(env, original, &infoOrig) != ANDROID_BITMAP_RESULT_SUCCESS) return -1;
    if (AndroidBitmap_getInfo(env, mask, &infoMask) != ANDROID_BITMAP_RESULT_SUCCESS) return -2;
    if (AndroidBitmap_getInfo(env, outBitmap, &infoOut) != ANDROID_BITMAP_RESULT_SUCCESS) return -3;

    if (infoOrig.width != infoMask.width || infoOrig.height != infoMask.height ||
        infoOrig.width != infoOut.width || infoOrig.height != infoOut.height) {
        return -4;
    }

    if (infoOrig.stride != infoOrig.width * 4 ||
        infoMask.stride != infoMask.width * 4 ||
        infoOut.stride != infoOut.width * 4) {
        return -6;
    }

    if (AndroidBitmap_lockPixels(env, original, &pixelsOrig) != ANDROID_BITMAP_RESULT_SUCCESS) return -7;
    if (AndroidBitmap_lockPixels(env, mask, &pixelsMask) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, original);
        return -8;
    }
    if (AndroidBitmap_lockPixels(env, outBitmap, &pixelsOut) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, mask);
        AndroidBitmap_unlockPixels(env, original);
        return -9;
    }

    cv::Mat srcMat(static_cast<int>(infoOrig.height), static_cast<int>(infoOrig.width), CV_8UC4, pixelsOrig);
    cv::Mat maskMat(static_cast<int>(infoMask.height), static_cast<int>(infoMask.width), CV_8UC4, pixelsMask);
    cv::Mat outMat(static_cast<int>(infoOut.height), static_cast<int>(infoOut.width), CV_8UC4, pixelsOut);

    srcMat.copyTo(outMat);

    cv::Mat grayMask;
    cv::cvtColor(maskMat, grayMask, cv::COLOR_RGBA2GRAY);

    int maxDim = std::max(static_cast<int>(infoOrig.width), static_cast<int>(infoOrig.height));
    int dilatePx = std::clamp(maxDim / 160, 5, 22);
    int inpaintRadius = std::clamp(maxDim / 240, 4, 15);

    cv::Mat dilatedMask1, dilatedMask2;
    cv::Mat elem1 = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(dilatePx * 2 + 1, dilatePx * 2 + 1));
    cv::dilate(grayMask, dilatedMask1, elem1);

    int edgeExpand = std::clamp(maxDim / 400, 2, 8);
    cv::Mat elem2 = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(edgeExpand * 2 + 1, edgeExpand * 2 + 1));
    cv::dilate(dilatedMask1, dilatedMask2, elem2);

    cv::Mat contourArea;
    cv::threshold(dilatedMask2, contourArea, 32, 255, cv::THRESH_BINARY);

    cv::Mat srcRgb;
    cv::cvtColor(srcMat, srcRgb, cv::COLOR_RGBA2RGB);

    // Primary engine: content-aware PatchMatch fill (real copied texture).
    // Falls back to the multi-scale diffusion fill if anything goes wrong.
    cv::Mat inpaintedRgb;
    try {
        inpaintedRgb = inpaint_engine::fillHole(srcRgb, contourArea);
    } catch (const cv::Exception& e) {
        LOGW("fillHole failed: %s", e.what());
        inpaintedRgb = inpaintHybrid(srcRgb, contourArea, inpaintRadius);
    }

    std::vector<cv::Mat> inRgba(4);
    cv::split(srcMat, inRgba);
    std::vector<cv::Mat> rgbChannels;
    cv::split(inpaintedRgb, rgbChannels);
    inRgba[0] = rgbChannels[0];
    inRgba[1] = rgbChannels[1];
    inRgba[2] = rgbChannels[2];
    cv::merge(inRgba, outMat);

    AndroidBitmap_unlockPixels(env, original);
    AndroidBitmap_unlockPixels(env, mask);
    AndroidBitmap_unlockPixels(env, outBitmap);

    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_magicimagepro_ml_NativeProcessor_seamlessComposite(
        JNIEnv* env,
        jobject /* this */,
        jobject original,
        jobject inpainted,
        jobject mask,
        jobject outBitmap) {

    AndroidBitmapInfo infoOrig, infoInp, infoMask, infoOut;
    void *pixelsOrig = nullptr, *pixelsInp = nullptr, *pixelsMask = nullptr, *pixelsOut = nullptr;

    if (AndroidBitmap_getInfo(env, original, &infoOrig) != ANDROID_BITMAP_RESULT_SUCCESS) return -1;
    if (AndroidBitmap_getInfo(env, inpainted, &infoInp) != ANDROID_BITMAP_RESULT_SUCCESS) return -2;
    if (AndroidBitmap_getInfo(env, mask, &infoMask) != ANDROID_BITMAP_RESULT_SUCCESS) return -3;
    if (AndroidBitmap_getInfo(env, outBitmap, &infoOut) != ANDROID_BITMAP_RESULT_SUCCESS) return -4;

    if (infoOrig.width != infoInp.width || infoOrig.height != infoInp.height ||
        infoOrig.width != infoMask.width || infoOrig.height != infoMask.height ||
        infoOrig.width != infoOut.width || infoOrig.height != infoOut.height) {
        return -5;
    }

    if (infoOrig.stride != infoOrig.width * 4 ||
        infoInp.stride != infoInp.width * 4 ||
        infoMask.stride != infoMask.width * 4 ||
        infoOut.stride != infoOut.width * 4) {
        return -6;
    }

    if (AndroidBitmap_lockPixels(env, original, &pixelsOrig) != ANDROID_BITMAP_RESULT_SUCCESS) return -7;
    if (AndroidBitmap_lockPixels(env, inpainted, &pixelsInp) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, original);
        return -8;
    }
    if (AndroidBitmap_lockPixels(env, mask, &pixelsMask) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, inpainted);
        AndroidBitmap_unlockPixels(env, original);
        return -9;
    }
    if (AndroidBitmap_lockPixels(env, outBitmap, &pixelsOut) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, mask);
        AndroidBitmap_unlockPixels(env, inpainted);
        AndroidBitmap_unlockPixels(env, original);
        return -10;
    }

    cv::Mat srcMat(static_cast<int>(infoOrig.height), static_cast<int>(infoOrig.width), CV_8UC4, pixelsOrig);
    cv::Mat inMat(static_cast<int>(infoInp.height), static_cast<int>(infoInp.width), CV_8UC4, pixelsInp);
    cv::Mat maskMat(static_cast<int>(infoMask.height), static_cast<int>(infoMask.width), CV_8UC4, pixelsMask);
    cv::Mat outMat(static_cast<int>(infoOut.height), static_cast<int>(infoOut.width), CV_8UC4, pixelsOut);

    cv::Mat srcRgb, inRgb, grayMask, binMask;
    cv::cvtColor(srcMat, srcRgb, cv::COLOR_RGBA2RGB);
    cv::cvtColor(inMat, inRgb, cv::COLOR_RGBA2RGB);
    cv::cvtColor(maskMat, grayMask, cv::COLOR_RGBA2GRAY);
    cv::threshold(grayMask, binMask, 127, 255, cv::THRESH_BINARY);

    cv::Mat finalRgb = seamlessCompositePipeline(srcRgb, inRgb, binMask);

    std::vector<cv::Mat> origRgbaChannels(4);
    cv::split(srcMat, origRgbaChannels);

    std::vector<cv::Mat> finalRgbChannels(3);
    cv::split(finalRgb, finalRgbChannels);

    std::vector<cv::Mat> outRgbaChannels = {
        finalRgbChannels[0],
        finalRgbChannels[1],
        finalRgbChannels[2],
        origRgbaChannels[3]
    };
    cv::merge(outRgbaChannels, outMat);

    AndroidBitmap_unlockPixels(env, original);
    AndroidBitmap_unlockPixels(env, inpainted);
    AndroidBitmap_unlockPixels(env, mask);
    AndroidBitmap_unlockPixels(env, outBitmap);

    return 0;
}
