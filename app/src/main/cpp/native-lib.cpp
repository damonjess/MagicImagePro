#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
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
    cv::Mat outTelea, outNs, combined;
    cv::inpaint(srcRgb, maskFull, outTelea, static_cast<double>(radius), cv::INPAINT_TELEA);
    cv::inpaint(srcRgb, maskFull, outNs, static_cast<double>(radius + 2), cv::INPAINT_NS);

    cv::Mat edgeMask;
    cv::Mat sobelX, sobelY;
    cv::Sobel(srcRgb, sobelX, CV_32F, 1, 0, 3);
    cv::Sobel(srcRgb, sobelY, CV_32F, 0, 1, 3);
    cv::magnitude(sobelX, sobelY, edgeMask);
    edgeMask = edgeMask / 255.0;

    cv::Mat blendW = cv::Mat::ones(srcRgb.size(), CV_32F) * 0.5f;
    for (int y = 0; y < srcRgb.rows; ++y) {
        for (int x = 0; x < srcRgb.cols; ++x) {
            if (maskFull.at<uchar>(y, x) > 0) {
                float e = (edgeMask.at<cv::Vec3f>(y, x)[0] +
                           edgeMask.at<cv::Vec3f>(y, x)[1] +
                           edgeMask.at<cv::Vec3f>(y, x)[2]) / 3.0f;
                if (e < 0.05f) {
                    blendW.at<float>(y, x) = 0.3f;
                } else if (e > 0.2f) {
                    blendW.at<float>(y, x) = 0.7f;
                }
            } else {
                blendW.at<float>(y, x) = 0.0f;
            }
        }
    }

    std::vector<cv::Mat> teleaCh, nsCh, outCh;
    cv::split(outTelea, teleaCh);
    cv::split(outNs, nsCh);
    outCh.resize(3);

    for (size_t c = 0; c < 3; ++c) {
        cv::Mat tF, nF;
        teleaCh[c].convertTo(tF, CV_32F);
        nsCh[c].convertTo(nF, CV_32F);
        cv::Mat one = cv::Mat::ones(blendW.size(), CV_32F);
        cv::Mat mixed = tF.mul(one - blendW) + nF.mul(blendW);
        mixed.convertTo(outCh[c], CV_8U);
    }
    cv::merge(outCh, combined);

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

    cv::Mat inpaintedRgb = inpaintHybrid(srcRgb, contourArea, inpaintRadius);

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
