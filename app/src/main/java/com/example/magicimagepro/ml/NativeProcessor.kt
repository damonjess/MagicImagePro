package com.example.magicimagepro.ml

import android.graphics.Bitmap

class NativeProcessor {
    companion object {
        init {
            System.loadLibrary("removal_engine")
        }
    }

    external fun processImage(original: Bitmap, mask: Bitmap, outBitmap: Bitmap): Int
    external fun seamlessComposite(original: Bitmap, inpainted: Bitmap, mask: Bitmap, outBitmap: Bitmap): Int

    /**
     * Polishes a generative fill: borrows the original's fine detail for the
     * masked area so a smooth model output gains the photo's real texture.
     */
    external fun refineFill(original: Bitmap, filled: Bitmap, mask: Bitmap, outBitmap: Bitmap): Int
}
