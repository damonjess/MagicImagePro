package com.example.magicimagepro.ml

import android.graphics.Bitmap

class NativeProcessor {

    external fun processImage(image: Bitmap, mask: Bitmap, result: Bitmap): Int

    companion object {
        init {
            try {
                System.loadLibrary("c++_shared")
            } catch (_: UnsatisfiedLinkError) {
                // Ignore if already loaded or statically linked
            }
            try {
                System.loadLibrary("opencv_java4")
            } catch (_: UnsatisfiedLinkError) {
                // Ignore if loaded automatically by linker
            }
            System.loadLibrary("removal_engine")
        }
    }
}
