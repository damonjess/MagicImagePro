package com.example.magicimagepro.ml

import android.graphics.Bitmap

class NativeProcessor {
    companion object {
        init {
            System.loadLibrary("removal_engine")
        }
    }

    external fun processImage(original: Bitmap, mask: Bitmap, outBitmap: Bitmap): Int
}
