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

static cv::Mat colorMatchBorder(const cv::Mat& srcRgb, const cv::Mat& inRgb, const cv::Mat& binMask) {
    int maxDim = std::max(srcRgb.cols, srcRgb.rows);
    int band = std::clamp(maxDim / 150, 4, 16);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(band * 2 + 1, band * 2 + 1));
    cv::Mat dilatedMask, erodedMask;
    cv::dilate(binMask, dilatedMask, element);
    cv::erode(binMask, erodedMask, element);

    cv::Mat outerBorder = dilatedMask & (~binMask);
    cv::Mat innerBorder = binMask & (~erodedMask);

    int outerCount = cv::countNonZero(outerBorder);
    int innerCount = cv::countNonZero(innerBorder);

    if (outerCount == 0 || innerCount == 0) {
        return inRgb.clone();
    }

    cv::Scalar meanSrc = cv::mean(srcRgb, outerBorder);
    cv::Scalar meanInp = cv::mean(inRgb, innerBorder);

    cv::Scalar gain, offset;
    bool useGain = true;

    for (int c = 0; c < 3; ++c) {
        offset[c] = meanSrc[c] - meanInp[c];
        if (meanInp[c] > 1.0) {
            gain[c] = meanSrc[c] / meanInp[c];
        } else {
            gain[c] = 1.0;
        }
        if (gain[c] < 0.6 || gain[c] > 1.4) {
            useGain = false;
        }
    }

    cv::Mat distMap;
    cv::distanceTransform(binMask, distMap, cv::DIST_L2, 3);

    const auto maxDist = static_cast<float>(band * 4);

    cv::Mat matched = inRgb.clone();
    for (int y = 0; y < inRgb.rows; ++y) {
        const auto* inPtr = inRgb.ptr<cv::Vec3b>(y);
        const uchar* mPtr = binMask.ptr<uchar>(y);
        const float* dPtr = distMap.ptr<float>(y);
        cv::Vec3b* outPtr = matched.ptr<cv::Vec3b>(y);

        for (int x = 0; x < inRgb.cols; ++x) {
            if (mPtr[x] > 0) {
                float d = dPtr[x];
                float t = std::clamp(d / maxDist, 0.0f, 1.0f);
                float weight = 1.0f - 0.5f * t;

                cv::Vec3b p = inPtr[x];
                for (int c = 0; c < 3; ++c) {
                    float val = static_cast<float>(p[c]);
                    if (useGain) {
                        float g = 1.0f + static_cast<float>(gain[c] - 1.0) * weight;
                        val *= g;
                    } else {
                        float off = static_cast<float>(offset[c]) * weight;
                        val += off;
                    }
                    outPtr[x][c] = cv::saturate_cast<uchar>(val);
                }
            }
        }
    }

    return matched;
}

static cv::Mat seamlessCompositePipeline(const cv::Mat& srcRgb,
                                          const cv::Mat& inRgb,
                                          const cv::Mat& binMask) {
    int maxDim = std::max(srcRgb.cols, srcRgb.rows);
    int featherPx = std::clamp(maxDim / 200, 4, 18);

    // Step 1: Color-match the inpainted border to the original pixels
    cv::Mat colorMatchedRgb = colorMatchBorder(srcRgb, inRgb, binMask);

    cv::Mat finalRgb;
    bool seamlessDone = false;

    if (cv::countNonZero(binMask) > 0) {
        // Step 2: cv::seamlessClone with NORMAL_CLONE
        try {
            cv::Mat cloneMask = binMask.clone();
            if (cloneMask.rows > 4 && cloneMask.cols > 4) {
                cloneMask.row(0).setTo(0);
                cloneMask.row(1).setTo(0);
                cloneMask.row(cloneMask.rows - 1).setTo(0);
                cloneMask.row(cloneMask.rows - 2).setTo(0);
                cloneMask.col(0).setTo(0);
                cloneMask.col(1).setTo(0);
                cloneMask.col(cloneMask.cols - 1).setTo(0);
                cloneMask.col(cloneMask.cols - 2).setTo(0);
            }

            if (cv::countNonZero(cloneMask) > 0) {
                cv::Point center(srcRgb.cols / 2, srcRgb.rows / 2);
                cv::Mat clonedResult;
                cv::seamlessClone(colorMatchedRgb, srcRgb, cloneMask, center, clonedResult, cv::NORMAL_CLONE);

                // Step 3: Feathered edge blend on cloned result
                featheredBlend(srcRgb, clonedResult, binMask, finalRgb, featherPx);
                seamlessDone = true;
            }
        } catch (const cv::Exception& e) {
            LOGW("cv::seamlessClone failed: %s", e.what());
        } catch (...) {
            LOGW("cv::seamlessClone threw unknown exception");
        }
    }

    // Step 4: Fall back to feathered-only if seamlessClone failed or threw
    if (!seamlessDone) {
        featheredBlend(srcRgb, colorMatchedRgb, binMask, finalRgb, featherPx);
    }

    return finalRgb;
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
