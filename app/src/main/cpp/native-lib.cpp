#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

static inline void featheredBlend(const cv::Mat& src, const cv::Mat& inpainted,
                                   const cv::Mat& dilatedMask, cv::Mat& dst,
                                   int featherPx) {
    cv::Mat blurredMask;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(std::max(3, featherPx * 2 + 1), std::max(3, featherPx * 2 + 1)));
    cv::dilate(dilatedMask, blurredMask, element);
    cv::GaussianBlur(blurredMask, blurredMask,
        cv::Size(std::max(3, featherPx * 4 + 1) | 1, std::max(3, featherPx * 4 + 1) | 1), 0.0);

    cv::Mat maskF, maskInv;
    blurredMask.convertTo(maskF, CV_32F, 1.0 / 255.0);
    maskInv = cv::Scalar(1.0f) - maskF;

    std::vector<cv::Mat> srcCh, inCh, outCh;
    cv::split(src, srcCh);
    cv::split(inpainted, inCh);
    outCh.resize(srcCh.size());

    for (size_t c = 0; c < srcCh.size(); ++c) {
        cv::Mat sF, iF;
        srcCh[c].convertTo(sF, CV_32F);
        inCh[c].convertTo(iF, CV_32F);
        cv::Mat blended = sF.mul(maskInv) + iF.mul(maskF);
        blended.convertTo(outCh[c], CV_8U);
    }
    cv::merge(outCh, dst);
}

static inline cv::Mat inpaintHybrid(const cv::Mat& srcRgb, const cv::Mat& maskFull,
                                     int radius, int featherPx) {
    cv::Mat outTelea, outNs, combined;
    cv::inpaint(srcRgb, maskFull, outTelea, static_cast<double>(radius), cv::INPAINT_TELEA);
    cv::inpaint(srcRgb, maskFull, outNs, static_cast<double>(radius + 2), cv::INPAINT_NS);

    cv::Mat edgeMask;
    cv::Mat sobelX, sobelY;
    cv::Sobel(srcRgb, sobelX, CV_32F, 1, 0, 3);
    cv::Sobel(srcRgb, sobelY, CV_32F, 0, 1, 3);
    cv::magnitude(sobelX, sobelY, edgeMask);
    edgeMask = edgeMask / 255.0;

    cv::Mat blendW = cv::Mat::ones(srcRgb.size(), CV_32F) * 0.55f;
    for (int y = 0; y < srcRgb.rows; ++y) {
        for (int x = 0; x < srcRgb.cols; ++x) {
            if (maskFull.at<uchar>(y, x) > 0) {
                float e = edgeMask.at<cv::Vec3f>(y, x)[0] +
                          edgeMask.at<cv::Vec3f>(y, x)[1] +
                          edgeMask.at<cv::Vec3f>(y, x)[2];
                e /= 3.0f;
                if (e < 0.05f) {
                    blendW.at<float>(y, x) = 0.35f;
                } else if (e > 0.2f) {
                    blendW.at<float>(y, x) = 0.75f;
                }
            } else {
                blendW.at<float>(y, x) = 0.0f;
            }
        }
    }

    std::vector<cv::Mat> teleaCh, nsCh, outCh;
    cv::split(outTelea, teleaCh);
    cv::split(outNs, nsCh);
    cv::split(srcRgb, outCh);

    for (size_t c = 0; c < 3; ++c) {
        cv::Mat tF, nF, sF;
        teleaCh[c].convertTo(tF, CV_32F);
        nsCh[c].convertTo(nF, CV_32F);
        outCh[c].convertTo(sF, CV_32F);
        cv::Mat one = cv::Mat::ones(blendW.size(), CV_32F);
        cv::Mat mixed = sF.mul(one - blendW) +
                        tF.mul(blendW * cv::Scalar(0.6f)) +
                        nF.mul(blendW * cv::Scalar(0.4f));
        mixed.convertTo(outCh[c], CV_8U);
    }
    cv::merge(outCh, combined);

    cv::Mat result;
    featheredBlend(srcRgb, combined, maskFull, result, featherPx);
    return result;
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

    if (infoOrig.stride != infoOrig.width * 4 ||
        infoMask.stride != infoMask.width * 4 ||
        infoOut.stride != infoOut.width * 4) {
        return -6;
    }

    AndroidBitmap_lockPixels(env, original, &pixelsOrig);
    AndroidBitmap_lockPixels(env, mask, &pixelsMask);
    AndroidBitmap_lockPixels(env, outBitmap, &pixelsOut);

    cv::Mat srcMat(infoOrig.height, infoOrig.width, CV_8UC4, pixelsOrig);
    cv::Mat maskMat(infoMask.height, infoMask.width, CV_8UC4, pixelsMask);
    cv::Mat outMat(infoOut.height, infoOut.width, CV_8UC4, pixelsOut);

    srcMat.copyTo(outMat);

    cv::Mat grayMask;
    cv::cvtColor(maskMat, grayMask, cv::COLOR_RGBA2GRAY);

    int maxDim = std::max(infoOrig.width, infoOrig.height);
    int dilatePx = std::clamp(maxDim / 160, 5, 22);
    int featherPx = std::clamp(maxDim / 200, 4, 18);
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

    cv::Mat inpaintedRgb = inpaintHybrid(srcRgb, contourArea, inpaintRadius, featherPx);

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
