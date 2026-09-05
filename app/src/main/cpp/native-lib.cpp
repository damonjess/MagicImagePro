#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

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

    // Ensure row strides match width * 4 (RGBA_8888) to prevent skewing
    if (infoOrig.stride != infoOrig.width * 4 ||
        infoMask.stride != infoMask.width * 4 ||
        infoOut.stride != infoOut.width * 4) {
        return -6;
    }

    AndroidBitmap_lockPixels(env, original, &pixelsOrig);
    AndroidBitmap_lockPixels(env, mask, &pixelsMask);
    AndroidBitmap_lockPixels(env, outBitmap, &pixelsOut);

    // Map raw bitmap memory directly to OpenCV matrices (Zero-Copy)
    cv::Mat srcMat(infoOrig.height, infoOrig.width, CV_8UC4, pixelsOrig);
    cv::Mat maskMat(infoMask.height, infoMask.width, CV_8UC4, pixelsMask);
    cv::Mat outMat(infoOut.height, infoOut.width, CV_8UC4, pixelsOut);

    // 1. Copy source image directly to output canvas as our base
    srcMat.copyTo(outMat);

    // 2. Convert mask to single-channel grayscale
    cv::Mat grayMask;
    cv::cvtColor(maskMat, grayMask, cv::COLOR_RGBA2GRAY);

    // 3. Dilate the mask slightly to cover edge shadows and borders
    cv::Mat dilatedMask;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::dilate(grayMask, dilatedMask, element);

    // 4. Run OpenCV's Fast Marching Method Inpainting
    // This fills the masked area using neighboring pixel textures smoothly
    cv::Mat srcRgb, outRgb;
    cv::cvtColor(srcMat, srcRgb, cv::COLOR_RGBA2RGB);
    cv::cvtColor(outMat, outRgb, cv::COLOR_RGBA2RGB);

    cv::inpaint(srcRgb, dilatedMask, outRgb, 3.0, cv::INPAINT_TELEA);

    // Convert back to RGBA and merge alpha channel into output
    cv::cvtColor(outRgb, outMat, cv::COLOR_RGB2RGBA);

    AndroidBitmap_unlockPixels(env, original);
    AndroidBitmap_unlockPixels(env, mask);
    AndroidBitmap_unlockPixels(env, outBitmap);

    return 0; // Success
}
