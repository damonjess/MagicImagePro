package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectRemover(context: Context) : TFLiteModel(context, "lama_dilated.tflite") {
    
    private val inputSize = 512 // LaMa uses 512x512
    
    fun removeObject(image: Bitmap, maskBitmap: Bitmap): Bitmap {
        // Resize inputs to model size
        val resizedImage = Bitmap.createScaledBitmap(image, inputSize, inputSize, true)
        val resizedMask = Bitmap.createScaledBitmap(maskBitmap, inputSize, inputSize, true)
        
        // Process image: [0, 255] -> [0, 1]
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        val tensorImage = TensorImage.fromBitmap(resizedImage)
        val processedImage = imageProcessor.process(tensorImage)
        
        // Process mask: binary 0 or 1
        val maskBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, inputSize, inputSize, 1), 
            org.tensorflow.lite.DataType.FLOAT32
        )
        val maskPixels = IntArray(inputSize * inputSize)
        resizedMask.getPixels(maskPixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        val maskFloats = FloatArray(inputSize * inputSize)
        for (i in maskPixels.indices) {
            // White in mask = area to remove (1.0), Black = keep (0.0)
            maskFloats[i] = if (Color.red(maskPixels[i]) > 128) 1.0f else 0.0f
        }
        maskBuffer.loadArray(maskFloats)
        
        // Run inference
        val outputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, inputSize, inputSize, 3),
            org.tensorflow.lite.DataType.FLOAT32
        )
        
        interpreter.runForMultipleInputsOutputs(
            arrayOf(processedImage.buffer, maskBuffer.buffer),
            mapOf(0 to outputBuffer.buffer)
        )
        
        // Convert output back to Bitmap
        return tensorBufferToBitmap(outputBuffer, inputSize).also {
            resizedImage.recycle()
            resizedMask.recycle()
        }
    }
    
    private fun tensorBufferToBitmap(buffer: TensorBuffer, size: Int): Bitmap {
        val output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val floats = buffer.floatArray
        
        val pixels = IntArray(size * size)
        for (i in 0 until size * size) {
            val r = (floats[i * 3].coerceIn(0f, 1f) * 255).toInt()
            val g = (floats[i * 3 + 1].coerceIn(0f, 1f) * 255).toInt()
            val b = (floats[i * 3 + 2].coerceIn(0f, 1f) * 255).toInt()
            pixels[i] = Color.rgb(r, g, b)
        }
        output.setPixels(pixels, 0, size, 0, 0, size, size)
        return output
    }
}
