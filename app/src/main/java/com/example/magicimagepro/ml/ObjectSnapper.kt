package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
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

        // 1. Calculate the Coordinate Prompt
        // We must translate where you tapped on your high-res image into where 
        // that point exists on the 1024x1024 model input.
        val scaleX = imageInputSize.toFloat() / originalWidth
        val scaleY = imageInputSize.toFloat() / originalHeight
        
        val modelTapX = tapX * scaleX
        val modelTapY = tapY * scaleY

        // 2. Prepare Input 1: The Image Tensor [1, 1024, 1024, 3] Float32
        val resizedImage = Bitmap.createScaledBitmap(image, imageInputSize, imageInputSize, true)
        val imageBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, imageInputSize, imageInputSize, 3), 
            DataType.FLOAT32
        )
        val imagePixels = IntArray(imageInputSize * imageInputSize)
        resizedImage.getPixels(imagePixels, 0, imageInputSize, 0, 0, imageInputSize, imageInputSize)
        
        val imageFloats = FloatArray(imageInputSize * imageInputSize * 3)
        for (i in imagePixels.indices) {
            val p = imagePixels[i]
            // Standard SAM normalization (0.0 to 1.0 or specific mean/std depending on your exact model)
            imageFloats[i * 3] = ((p shr 16) and 0xFF) / 255.0f
            imageFloats[i * 3 + 1] = ((p shr 8) and 0xFF) / 255.0f
            imageFloats[i * 3 + 2] = (p and 0xFF) / 255.0f
        }
        imageBuffer.loadArray(imageFloats)
        resizedImage.recycle()

        // 3. Prepare Input 2: Point Coordinates Tensor [1, 1, 2] Float32
        // Shape is [Batch, NumPoints, (X, Y)]
        val coordsBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1, 2), DataType.FLOAT32)
        coordsBuffer.loadArray(floatArrayOf(modelTapX, modelTapY))

        // 4. Prepare Input 3: Point Labels Tensor [1, 1] Float32
        // 1.0 means "Foreground" (select this). 0.0 means "Background" (ignore this).
        val labelsBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 1), DataType.FLOAT32)
        labelsBuffer.loadArray(floatArrayOf(1.0f))

        // 5. Prepare Output: The Mask Tensor [1, 1, 256, 256] Float32
        val outputMaskBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, 1, maskOutputSize, maskOutputSize), 
            DataType.FLOAT32
        )

        // 6. Run Inference
        val inputs = arrayOf(imageBuffer.buffer, coordsBuffer.buffer, labelsBuffer.buffer)
        val outputs = mapOf(0 to outputMaskBuffer.buffer)
        
        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        // 7. Parse output floats into a visual mask Bitmap
        val rawMaskBitmap = parseMask(outputMaskBuffer, maskOutputSize)

        // 8. Scale the small 256x256 mask back up to perfectly fit your original photo
        val finalMask = Bitmap.createScaledBitmap(rawMaskBitmap, originalWidth, originalHeight, true)
        rawMaskBitmap.recycle()

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
