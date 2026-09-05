package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectSnapper(context: Context, modelPath: String) : TFLiteModel(context, modelPath) {

    // NOTE: Check your specific SAM TFLite model's input/output shapes. 
    // These are standard for EdgeSAM/MobileSAM architectures.
    private val imageInputSize = 1024 
    private val maskOutputSize = 256

    /**
     * @param image The original high-res Bitmap
     * @param tapX The X coordinate of where the user tapped (relative to the original Bitmap)
     * @param tapY The Y coordinate of where the user tapped (relative to the original Bitmap)
     * @return A snapping mask Bitmap matching the original image dimensions
     */
    fun generateMask(image: Bitmap, tapX: Float, tapY: Float): Bitmap {
        val originalWidth = image.width
        val originalHeight = image.height

        // 1. Uniform scale that fits the image inside imageInputSize, preserving aspect ratio
        val scale = minOf(
            imageInputSize.toFloat() / originalWidth,
            imageInputSize.toFloat() / originalHeight
        )
        val scaledW = (originalWidth * scale).toInt()
        val scaledH = (originalHeight * scale).toInt()
        val padX = (imageInputSize - scaledW) / 2
        val padY = (imageInputSize - scaledH) / 2

        // 2. Draw the scaled image onto a black square canvas (letterbox, not stretch)
        val squareBitmap = Bitmap.createBitmap(imageInputSize, imageInputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(squareBitmap)
        canvas.drawColor(Color.BLACK)
        val scaledImage = Bitmap.createScaledBitmap(image, scaledW, scaledH, true)
        canvas.drawBitmap(scaledImage, padX.toFloat(), padY.toFloat(), null)
        scaledImage.recycle()

        // 3. Tap coordinate uses the SAME scale + offset as the image
        val modelTapX = tapX * scale + padX
        val modelTapY = tapY * scale + padY

        val imageBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, imageInputSize, imageInputSize, 3), DataType.FLOAT32
        )
        val imagePixels = IntArray(imageInputSize * imageInputSize)
        squareBitmap.getPixels(imagePixels, 0, imageInputSize, 0, 0, imageInputSize, imageInputSize)
        val imageFloats = FloatArray(imageInputSize * imageInputSize * 3)
        for (i in imagePixels.indices) {
            val p = imagePixels[i]
            imageFloats[i * 3] = ((p shr 16) and 0xFF) / 255.0f
            imageFloats[i * 3 + 1] = ((p shr 8) and 0xFF) / 255.0f
            imageFloats[i * 3 + 2] = (p and 0xFF) / 255.0f
        }
        imageBuffer.loadArray(imageFloats)
        squareBitmap.recycle()

        val coordsBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1, 2), DataType.FLOAT32)
        coordsBuffer.loadArray(floatArrayOf(modelTapX, modelTapY))
        val labelsBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
        labelsBuffer.loadArray(floatArrayOf(1.0f))
        val outputMaskBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, 1, maskOutputSize, maskOutputSize), DataType.FLOAT32
        )

        interpreter.runForMultipleInputsOutputs(
            arrayOf(imageBuffer.buffer, coordsBuffer.buffer, labelsBuffer.buffer),
            mapOf(0 to outputMaskBuffer.buffer)
        )

        val rawMaskBitmap = parseMask(outputMaskBuffer, maskOutputSize)

        // 4. Crop out the padding in mask space (same ratio as the input padding) before scaling up
        val maskScale = maskOutputSize.toFloat() / imageInputSize
        val cropX = (padX * maskScale).toInt()
        val cropY = (padY * maskScale).toInt()
        val cropW = (scaledW * maskScale).toInt().coerceAtLeast(1)
        val cropH = (scaledH * maskScale).toInt().coerceAtLeast(1)

        val croppedMask = Bitmap.createBitmap(rawMaskBitmap, cropX, cropY, cropW, cropH)
        rawMaskBitmap.recycle()

        val finalMask = Bitmap.createScaledBitmap(croppedMask, originalWidth, originalHeight, true)
        croppedMask.recycle()

        return finalMask
    }

    private fun parseMask(buffer: TensorBuffer, size: Int): Bitmap {
        val maskBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val floats = buffer.floatArray
        val pixels = IntArray(size * size)
        
        // SAM usually outputs "logits" (raw confident scores). 
        // A common threshold is 0.0. Anything above 0 is the object, below 0 is background.
        val threshold = 0.0f 

        for (i in 0 until size * size) {
            if (floats[i] > threshold) {
                // Foreground (Object): Make it solid white so your ObjectRemover can read it
                pixels[i] = Color.WHITE
            } else {
                // Background: Keep it black
                pixels[i] = Color.BLACK
            }
        }
        
        maskBitmap.setPixels(pixels, 0, size, 0, 0, size, size)
        return maskBitmap
    }
}
