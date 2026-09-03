#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#define LOG_TAG "NativeProcessor"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jint JNICALL
Java_com_example_magicimagepro_ml_NativeProcessor_processImage(
        JNIEnv* env,
        jobject thiz,
        jobject origBitmap,
        jobject maskBitmap,
        jobject outBitmap) {

    AndroidBitmapInfo infoOrig;
    AndroidBitmapInfo infoMask;
    AndroidBitmapInfo infoOut;

    if (AndroidBitmap_getInfo(env, origBitmap, &infoOrig) < 0 ||
        AndroidBitmap_getInfo(env, maskBitmap, &infoMask) < 0 ||
        AndroidBitmap_getInfo(env, outBitmap, &infoOut) < 0) {
        return -1;
    }

    if (infoOrig.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        infoMask.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        infoOut.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return -2;
    }

    if (infoOrig.width != infoMask.width || infoOrig.height != infoMask.height ||
        infoOrig.width != infoOut.width || infoOrig.height != infoOut.height) {
        return -3;
    }

    // Ensure row strides match width * 4 (RGBA_8888) to prevent skewing
    if (infoOrig.stride != infoOrig.width * 4 ||
        infoMask.stride != infoMask.width * 4 ||
        infoOut.stride != infoOut.width * 4) {
        return -6;
    }

    void* pixelsOrig = nullptr;
    void* pixelsMask = nullptr;
    void* pixelsOut = nullptr;

    if (AndroidBitmap_lockPixels(env, origBitmap, &pixelsOrig) < 0) return -4;
    if (AndroidBitmap_lockPixels(env, maskBitmap, &pixelsMask) < 0) {
        AndroidBitmap_unlockPixels(env, origBitmap);
        return -4;
    }
    if (AndroidBitmap_lockPixels(env, outBitmap, &pixelsOut) < 0) {
        AndroidBitmap_unlockPixels(env, origBitmap);
        AndroidBitmap_unlockPixels(env, maskBitmap);
        return -4;
    }

    uint32_t* srcPixels = static_cast<uint32_t*>(pixelsOrig);
    uint32_t* maskPixels = static_cast<uint32_t*>(pixelsMask);
    uint32_t* dstPixels = static_cast<uint32_t*>(pixelsOut);

    int width = infoOrig.width;
    int height = infoOrig.height;
    int totalPixels = width * height;

    for (int i = 0; i < totalPixels; ++i) {
        uint32_t maskPixel = maskPixels[i];
        uint8_t maskAlpha = (maskPixel >> 24) & 0xFF;
        uint8_t maskRed = (maskPixel >> 16) & 0xFF;
        uint8_t maskGreen = (maskPixel >> 8) & 0xFF;
        uint8_t maskBlue = maskPixel & 0xFF;

        if (maskAlpha > 0 || maskRed > 0 || maskGreen > 0 || maskBlue > 0) {
            // Render a clean, precise red block over painted strokes
            dstPixels[i] = 0xFF0000FF; // Red in RGBA_8888
        } else {
            dstPixels[i] = srcPixels[i];
        }
    }

    AndroidBitmap_unlockPixels(env, origBitmap);
    AndroidBitmap_unlockPixels(env, maskBitmap);
    AndroidBitmap_unlockPixels(env, outBitmap);

    return 0;
}
