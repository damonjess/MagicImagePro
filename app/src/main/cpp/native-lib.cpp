#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>

#define LOG_TAG "RemovalEngine"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jint JNICALL
Java_com_example_magicimagepro_ml_NativeProcessor_processImage(
        JNIEnv* env, jobject /* this */,
        jobject original, jobject mask, jobject outBitmap) {

    if (!original || !mask || !outBitmap) {
        LOGE("Null bitmap handle passed to processImage");
        return -1;
    }

    AndroidBitmapInfo infoOrig, infoMask, infoOut;
    if (AndroidBitmap_getInfo(env, original, &infoOrig) != ANDROID_BITMAP_RESULT_SUCCESS) return -1;
    if (AndroidBitmap_getInfo(env, mask, &infoMask) != ANDROID_BITMAP_RESULT_SUCCESS) return -2;
    if (AndroidBitmap_getInfo(env, outBitmap, &infoOut) != ANDROID_BITMAP_RESULT_SUCCESS) return -3;

    if (infoOrig.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        infoMask.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        infoOut.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Unsupported bitmap format. Bitmaps must be RGBA_8888");
        return -4;
    }

    void *pixelsOrig = nullptr;
    void *pixelsMask = nullptr;
    void *pixelsOut = nullptr;

    if (AndroidBitmap_lockPixels(env, original, &pixelsOrig) != ANDROID_BITMAP_RESULT_SUCCESS || !pixelsOrig) {
        LOGE("Failed to lock original bitmap pixels");
        return -5;
    }
    if (AndroidBitmap_lockPixels(env, mask, &pixelsMask) != ANDROID_BITMAP_RESULT_SUCCESS || !pixelsMask) {
        LOGE("Failed to lock mask bitmap pixels");
        AndroidBitmap_unlockPixels(env, original);
        return -6;
    }
    if (AndroidBitmap_lockPixels(env, outBitmap, &pixelsOut) != ANDROID_BITMAP_RESULT_SUCCESS || !pixelsOut) {
        LOGE("Failed to lock outBitmap pixels");
        AndroidBitmap_unlockPixels(env, original);
        AndroidBitmap_unlockPixels(env, mask);
        return -7;
    }

    jint resultCode = 0;

    try {
        // Construct OpenCV matrices using explicit row stride (step)
        cv::Mat matOrig(infoOrig.height, infoOrig.width, CV_8UC4, pixelsOrig, infoOrig.stride);
        cv::Mat matMask(infoMask.height, infoMask.width, CV_8UC4, pixelsMask, infoMask.stride);
        cv::Mat matOut(infoOut.height, infoOut.width, CV_8UC4, pixelsOut, infoOut.stride);

        // Extract single-channel mask.
        // Channel 0 (Red) is 255 for White mask strokes on Black background.
        cv::Mat ch0, ch3, singleChannelMask;
        cv::extractChannel(matMask, ch0, 0);
        cv::extractChannel(matMask, ch3, 3);

        int nonZero0 = cv::countNonZero(ch0);
        if (nonZero0 > 0) {
            cv::threshold(ch0, singleChannelMask, 50, 255, cv::THRESH_BINARY);
        } else {
            // Fallback for alpha-channel masks
            cv::threshold(ch3, singleChannelMask, 50, 255, cv::THRESH_BINARY);
        }

        int maskPixelsCount = cv::countNonZero(singleChannelMask);
        LOGI("Mask non-zero pixels: %d", maskPixelsCount);

        if (maskPixelsCount == 0) {
            // No mask drawn, simply copy original image to output
            matOrig.copyTo(matOut);
        } else {
            // Circular Dilation to expand mask over object edges and shadows
            int dilationSize = 8;
            cv::Mat structuralElement = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1),
                    cv::Point(dilationSize, dilationSize)
            );

            cv::Mat dilatedMask;
            cv::dilate(singleChannelMask, dilatedMask, structuralElement);

            // Inpaint: Convert RGBA to 3-channel BGR, execute cv::inpaint, then convert back to RGBA
            cv::Mat bgrOrig, bgrInpainted;
            cv::cvtColor(matOrig, bgrOrig, cv::COLOR_RGBA2BGR);

            // Execute Telea inpainting algorithm with a 3px neighborhood radius
            cv::inpaint(bgrOrig, dilatedMask, bgrInpainted, 3.0, cv::INPAINT_TELEA);

            // Write back to the output bitmap in RGBA format
            cv::cvtColor(bgrInpainted, matOut, cv::COLOR_BGR2RGBA);
        }

    } catch (const cv::Exception& e) {
        LOGE("OpenCV Exception: %s", e.what());
        resultCode = -10;
    } catch (const std::exception& e) {
        LOGE("Standard Exception: %s", e.what());
        resultCode = -11;
    } catch (...) {
        LOGE("Unknown Native Exception caught");
        resultCode = -12;
    }

    // Always safely unlock bitmap pixels
    AndroidBitmap_unlockPixels(env, original);
    AndroidBitmap_unlockPixels(env, mask);
    AndroidBitmap_unlockPixels(env, outBitmap);

    return resultCode;
}
