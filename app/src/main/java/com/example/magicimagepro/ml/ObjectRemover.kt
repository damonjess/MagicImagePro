package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ObjectRemover(context: Context) : TFLiteModel(context, "lama_dilated.tflite") {
    
    private val modelWidth = 512
    private val modelHeight = 512
    
    suspend fun removeObject(image: Bitmap, mask: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        // 1. Pad the original rectangular images into perfect squares
        val paddedImage = padToSquare(image)
        val paddedMask = padToSquare(mask)

        // 2. Scale the squares safely down to the 512x512 tensor requirement
        val scaledImage = Bitmap.createScaledBitmap(paddedImage, modelWidth, modelHeight, true)
        val scaledMask = Bitmap.createScaledBitmap(paddedMask, modelWidth, modelHeight, true)

        // Detect input tensor indices and formats (NCHW vs NHWC)
        var imageInputIndex = 0
        var maskInputIndex = 1
        var isNCHW = false

        for (i in 0 until interpreter.inputTensorCount) {
            val shape = interpreter.getInputTensor(i).shape()
            val channels = if (shape.size == 4) { if (shape[1] == 3 || shape[1] == 1) shape[1] else shape[3] } else 3
            if (channels == 3) {
                imageInputIndex = i
                if (shape.size == 4 && shape[1] == 3) isNCHW = true
            } else if (channels == 1) {
                maskInputIndex = i
            }
        }

        val outShape = interpreter.getOutputTensor(0).shape()
        val outIsNCHW = outShape.size == 4 && outShape[1] == 3

        val imageBuffer = bitmapToByteBuffer(scaledImage, isMask = false, isNCHW = isNCHW)
        val maskBuffer = bitmapToByteBuffer(scaledMask, isMask = true, isNCHW = false)

        val inputs = arrayOfNulls<Any>(2)
        inputs[imageInputIndex] = imageBuffer
        inputs[maskInputIndex] = maskBuffer

        val outputBuffer: Any = if (outIsNCHW) {
            Array(1) { Array(3) { Array(modelHeight) { FloatArray(modelWidth) } } }
        } else {
            Array(1) { Array(modelHeight) { Array(modelWidth) { FloatArray(3) } } }
        }
        
        interpreter.runForMultipleInputsOutputs(inputs, mapOf(0 to outputBuffer))

        // 3. Convert tensor to 512x512 Bitmap
        val rawAiOutput = convertOutputToBitmap(outputBuffer, modelWidth, modelHeight, outIsNCHW)

        // 4. Scale it back up to the massive square size, THEN crop the borders off
        val inpaintedPadded = Bitmap.createScaledBitmap(rawAiOutput, paddedImage.width, paddedImage.height, true)
        val inpaintedFull = cropFromSquare(inpaintedPadded, image.width, image.height)

        // 5. Run custom alpha-blending composite to hide the seam!
        val softMask = blurMask(mask, blurRadius = 6f)

        val resultBitmap = Bitmap.createBitmap(image.width, image.height, image.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(image, 0f, 0f, null)

        val blendBitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        val patchCanvas = Canvas(blendBitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        patchCanvas.drawBitmap(softMask, 0f, 0f, null)
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
        patchCanvas.drawBitmap(inpaintedFull, 0f, 0f, paint)

        canvas.drawBitmap(blendBitmap, 0f, 0f, null)

        // Cleanup
        paddedImage.recycle()
        paddedMask.recycle()
        scaledImage.recycle()
        scaledMask.recycle()
        rawAiOutput.recycle()
        inpaintedPadded.recycle()
        inpaintedFull.recycle()
        softMask.recycle()
        blendBitmap.recycle()

        resultBitmap
    }

    // --- Helper Methods ---

    private fun padToSquare(bitmap: Bitmap): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        if (bitmap.width == maxDim && bitmap.height == maxDim) return bitmap

        // Create a square canvas and draw the original image exactly in the center
        val padded = Bitmap.createBitmap(maxDim, maxDim, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawColor(Color.BLACK) // The model interprets black as empty space

        val left = (maxDim - bitmap.width) / 2f
        val top = (maxDim - bitmap.height) / 2f
        canvas.drawBitmap(bitmap, left, top, null)
        
        return padded
    }

    private fun cropFromSquare(squaredBitmap: Bitmap, originalWidth: Int, originalHeight: Int): Bitmap {
        if (squaredBitmap.width == originalWidth && squaredBitmap.height == originalHeight) return squaredBitmap

        val left = (squaredBitmap.width - originalWidth) / 2
        val top = (squaredBitmap.height - originalHeight) / 2
        
        return Bitmap.createBitmap(squaredBitmap, left, top, originalWidth, originalHeight)
    }

    private fun blurMask(mask: Bitmap, blurRadius: Float): Bitmap {
        // The incoming mask is Black (0xFF000000) for background, White (0xFFFFFFFF) for mask.
        // We need to convert it so that black is transparent, and white is opaque.
        val alphaSource = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(mask.width * mask.height)
        mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)
        for (i in pixels.indices) {
            val r = Color.red(pixels[i])
            pixels[i] = Color.argb(r, 255, 255, 255)
        }
        alphaSource.setPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)

        val outputBitmap = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        // Using BlurMaskFilter instead of deprecated RenderScript
        paint.maskFilter = BlurMaskFilter(blurRadius.coerceIn(1f, 25f), BlurMaskFilter.Blur.NORMAL)
        
        // Extract the alpha channel from the source mask to apply the blur filter correctly
        val alphaMask = alphaSource.extractAlpha(paint, null)
        
        // Draw the blurred alpha mask in white onto our output bitmap
        val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG)
        drawPaint.color = Color.WHITE
        canvas.drawBitmap(alphaMask, 0f, 0f, drawPaint)
        
        alphaMask.recycle()
        alphaSource.recycle()
        return outputBitmap
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap, isMask: Boolean, isNCHW: Boolean): ByteBuffer {
        val channels = if (isMask) 1 else 3
        val byteBuffer = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * channels)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(modelWidth * modelHeight)
        bitmap.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight)
        val area = modelWidth * modelHeight

        if (isNCHW && !isMask) {
            val rFloats = FloatArray(area)
            val gFloats = FloatArray(area)
            val bFloats = FloatArray(area)
            for (i in pixels.indices) {
                val p = pixels[i]
                rFloats[i] = ((p shr 16) and 0xFF) / 255.0f
                gFloats[i] = ((p shr 8) and 0xFF) / 255.0f
                bFloats[i] = (p and 0xFF) / 255.0f
            }
            for (f in rFloats) byteBuffer.putFloat(f)
            for (f in gFloats) byteBuffer.putFloat(f)
            for (f in bFloats) byteBuffer.putFloat(f)
        } else {
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255.0f
                val g = ((p shr 8) and 0xFF) / 255.0f
                val b = (p and 0xFF) / 255.0f
                if (isMask) {
                    byteBuffer.putFloat(r)
                } else {
                    byteBuffer.putFloat(r)
                    byteBuffer.putFloat(g)
                    byteBuffer.putFloat(b)
                }
            }
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun convertOutputToBitmap(outputBuffer: Any, width: Int, height: Int, isNCHW: Boolean): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        val outNCHW = if (isNCHW) outputBuffer as Array<Array<Array<FloatArray>>> else null
        val outNHWC = if (!isNCHW) outputBuffer as Array<Array<Array<FloatArray>>> else null

        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = y * width + x
                val rF: Float
                val gF: Float
                val bF: Float
                if (isNCHW) {
                    rF = outNCHW!![0][0][y][x]
                    gF = outNCHW[0][1][y][x]
                    bF = outNCHW[0][2][y][x]
                } else {
                    rF = outNHWC!![0][y][x][0]
                    gF = outNHWC[0][y][x][1]
                    bF = outNHWC[0][y][x][2]
                }
                val r = (rF.coerceIn(0f, 1f) * 255).toInt()
                val g = (gF.coerceIn(0f, 1f) * 255).toInt()
                val b = (bF.coerceIn(0f, 1f) * 255).toInt()
                pixels[i] = Color.rgb(r, g, b)
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
