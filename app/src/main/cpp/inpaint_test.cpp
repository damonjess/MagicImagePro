//
// Desktop test harness for inpaint_engine.h.
// Runs the content-aware fill on a real photo + mask and writes
// side-by-side results for visual inspection.
//
// Usage: ./inpaint_test <image> <mask> <out_prefix> [initImage]
//
// When <initImage> is given it is treated as an externally produced fill of
// the same mask (e.g. a LaMa output) and re-textured by the engine.
//
#include "inpaint_engine.h"

#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <image> <mask> <out_prefix>\n", argv[0]);
        return 1;
    }
    cv::Mat img = cv::imread(argv[1]);
    cv::Mat mask = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (img.empty() || mask.empty()) {
        std::printf("failed to load inputs\n");
        return 1;
    }
    if (mask.size() != img.size()) {
        cv::resize(mask, mask, img.size(), 0, 0, cv::INTER_NEAREST);
    }

    std::printf("image %dx%d, mask holes = %d px\n",
                img.cols, img.rows, cv::countNonZero(mask));

    // Baseline for comparison: what the app used to ship (Telea/NS smear).
    cv::Mat telea;
    cv::inpaint(img, mask, telea, 10.0, cv::INPAINT_TELEA);

    double t0 = (double)cv::getTickCount();
    cv::Mat filled = inpaint_engine::fillHole(img, mask);
    double t1 = (double)cv::getTickCount();
    std::printf("fillHole: %.2f s\n", (t1 - t0) / cv::getTickFrequency());

    // Downscale composite for viewing: [ original | telea | new engine ]
    cv::Mat oS, tS, fS;
    const int W = 500;
    double sc = (double)W / std::max(img.cols, img.rows);
    cv::resize(img, oS, cv::Size(), sc, sc);
    cv::resize(telea, tS, cv::Size(), sc, sc);
    cv::resize(filled, fS, cv::Size(), sc, sc);
    cv::Mat canvas(cv::Size(W * 3 + 20, oS.rows), CV_8UC3, cv::Scalar(30, 30, 30));
    oS.copyTo(canvas(cv::Rect(0, 0, oS.cols, oS.rows)));
    tS.copyTo(canvas(cv::Rect(W + 10, 0, tS.cols, tS.rows)));
    fS.copyTo(canvas(cv::Rect(2 * W + 20, 0, fS.cols, fS.rows)));
    cv::imwrite(std::string(argv[3]) + "_compare.png", canvas);
    cv::imwrite(std::string(argv[3]) + "_filled.png", filled);

    std::printf("wrote %s_compare.png and %s_filled.png\n", argv[3], argv[3]);

    if (argc >= 5) {
        cv::Mat init = cv::imread(argv[4]);
        if (init.empty()) {
            std::printf("failed to load init image %s\n", argv[4]);
            return 1;
        }
        if (init.size() != img.size()) {
            cv::resize(init, init, img.size(), 0, 0, cv::INTER_CUBIC);
        }
        double t2 = (double)cv::getTickCount();
        cv::Mat refined = inpaint_engine::refineFill(img, mask, init);
        double t3 = (double)cv::getTickCount();
        std::printf("refineFill: %.2f s\n", (t3 - t2) / cv::getTickFrequency());
        cv::imwrite(std::string(argv[3]) + "_refined.png", refined);

        cv::Mat iS, rS;
        cv::resize(init, iS, cv::Size(), sc, sc);
        cv::resize(refined, rS, cv::Size(), sc, sc);
        cv::Mat canvas2(cv::Size(W * 4 + 30, oS.rows), CV_8UC3, cv::Scalar(30, 30, 30));
        oS.copyTo(canvas2(cv::Rect(0, 0, oS.cols, oS.rows)));
        tS.copyTo(canvas2(cv::Rect(W + 10, 0, tS.cols, tS.rows)));
        iS.copyTo(canvas2(cv::Rect(2 * W + 20, 0, iS.cols, iS.rows)));
        rS.copyTo(canvas2(cv::Rect(3 * W + 30, 0, rS.cols, rS.rows)));
        cv::imwrite(std::string(argv[3]) + "_compare4.png", canvas2);
        std::printf("wrote %s_refined.png and %s_compare4.png\n", argv[3], argv[3]);
    }
    return 0;
}
