package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ImageUpscaler(context: Context) : TFLiteModel(context, "real_esrgan_x4plus.tflite") {
    
    // ESRGAN usually outputs 4x resolution
    private val scaleFactor = 4
    
    fun upscale(image: Bitmap): Bitmap {
        val inputSize = 256 // Process in tiles for memory efficiency
        val outputSize = inputSize * scaleFactor
        
        // For simplicity, resize to inputSize (production: use tiling for large images)
        val resized = Bitmap.createScaledBitmap(image, inputSize, inputSize, true)
        
        // Convert to float array [0, 1]
        val inputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, inputSize, inputSize, 3),
            org.tensorflow.lite.DataType.FLOAT32
        )
        
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        val floatArray = FloatArray(inputSize * inputSize * 3)
        for (i in pixels.indices) {
            floatArray[i * 3] = Color.red(pixels[i]) / 255.0f
            floatArray[i * 3 + 1] = Color.green(pixels[i]) / 255.0f
            floatArray[i * 3 + 2] = Color.blue(pixels[i]) / 255.0f
        }
        inputBuffer.loadArray(floatArray)
        
        // Run inference
        val outputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, outputSize, outputSize, 3),
            org.tensorflow.lite.DataType.FLOAT32
        )
        
        interpreter.run(inputBuffer.buffer, outputBuffer.buffer)
        
        // Convert back
        return bufferToBitmap(outputBuffer, outputSize).also {
            resized.recycle()
        }
    }
    
    private fun bufferToBitmap(buffer: TensorBuffer, size: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val floats = buffer.floatArray
        
        val pixels = IntArray(size * size)
        for (i in 0 until size * size) {
            val r = (floats[i * 3].coerceIn(0f, 1f) * 255).toInt()
            val g = (floats[i * 3 + 1].coerceIn(0f, 1f) * 255).toInt()
            val b = (floats[i * 3 + 2].coerceIn(0f, 1f) * 255).toInt()
            pixels[i] = Color.rgb(r, g, b)
        }
        bitmap.setPixels(pixels, 0, size, 0, 0, size, size)
        return bitmap
    }
}
