package com.example.magicimagepro.ml

import android.graphics.Bitmap

class NativeProcessor {

    external fun processImage(image: Bitmap, mask: Bitmap, result: Bitmap): Int

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }
}
