package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ObjectRemover(context: Context) : TFLiteModel(context, "lama_dilated.tflite") {
    
    private val inputSize = 512
    
    suspend fun removeObject(image: Bitmap, maskBitmap: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        val bounds = getMaskBounds(maskBitmap) ?: return@withContext image 
        val centerX = bounds.centerX()
        val centerY = bounds.centerY()
        
        // 1. Dynamic Context Multiplier
        val maxDim = maxOf(bounds.width(), bounds.height())
        val maxImageDim = maxOf(image.width, image.height)
        val objectRatio = maxDim.toFloat() / maxImageDim.toFloat()

        val paddingMultiplier = when {
            objectRatio > 0.35f -> 3.2f  
            objectRatio > 0.15f -> 2.4f  
            else -> 1.8f                 
        }
        
        val cropSize = (maxDim * paddingMultiplier).toInt().coerceAtLeast(inputSize)
        
        val imageCrop = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        val maskCrop = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        val dx = cropSize / 2f - centerX
        val dy = cropSize / 2f - centerY
        
        Canvas(imageCrop).apply { drawColor(Color.BLACK); drawBitmap(image, dx, dy, null) }
        Canvas(maskCrop).apply { drawColor(Color.BLACK); drawBitmap(maskBitmap, dx, dy, null) }
        
        val resizedImg = Bitmap.createScaledBitmap(imageCrop, inputSize, inputSize, true)
        val resizedMask = Bitmap.createScaledBitmap(maskCrop, inputSize, inputSize, true)
        
        // 2. Dynamic Input/Output Detection
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
        
        val imageShape = interpreter.getInputTensor(imageInputIndex).shape()
        val maskShape = interpreter.getInputTensor(maskInputIndex).shape()
        val outShape = interpreter.getOutputTensor(0).shape()
        val outIsNCHW = outShape.size == 4 && outShape[1] == 3
        
        val inputImageBuffer = TensorBuffer.createFixedSize(imageShape, DataType.FLOAT32)
        val maskBuffer = TensorBuffer.createFixedSize(maskShape, DataType.FLOAT32)
        val outputBuffer = TensorBuffer.createFixedSize(outShape, DataType.FLOAT32)
        
        val imagePixels = IntArray(inputSize * inputSize)
        val maskPixels = IntArray(inputSize * inputSize)
        resizedImg.getPixels(imagePixels, 0, inputSize, 0, 0, inputSize, inputSize)
        resizedMask.getPixels(maskPixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        // 3. Circular Mask Dilation
        val dilationRadius = 8 
        val dilatedMaskPixels = BooleanArray(inputSize * inputSize)
        
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                if (((maskPixels[y * inputSize + x] shr 16) and 0xFF) > 100) {
                    for (dy in -dilationRadius..dilationRadius) {
                        for (dx in -dilationRadius..dilationRadius) {
                            if (dx * dx + dy * dy <= dilationRadius * dilationRadius) {
                                val nx = x + dx
                                val ny = y + dy
                                if (nx in 0 until inputSize && ny in 0 until inputSize) {
                                    dilatedMaskPixels[ny * inputSize + nx] = true
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 4. Fill Tensors
        val imageFloats = FloatArray(inputSize * inputSize * 3)
        val maskFloats = FloatArray(inputSize * inputSize)
        val area = inputSize * inputSize
        
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val i = y * inputSize + x
                val isMask = dilatedMaskPixels[i]
                maskFloats[i] = if (isMask) 1.0f else 0.0f 
                
                val p = imagePixels[i]
                val r = if (isMask) 0f else ((p shr 16) and 0xFF) / 255.0f
                val g = if (isMask) 0f else ((p shr 8) and 0xFF) / 255.0f
                val b = if (isMask) 0f else (p and 0xFF) / 255.0f
                
                if (isNCHW) {
                    imageFloats[0 * area + i] = r
                    imageFloats[1 * area + i] = g
                    imageFloats[2 * area + i] = b
                } else {
                    imageFloats[i * 3 + 0] = r
                    imageFloats[i * 3 + 1] = g
                    imageFloats[i * 3 + 2] = b
                }
            }
        }
        
        inputImageBuffer.loadArray(imageFloats)
        maskBuffer.loadArray(maskFloats)
        
        // 5. Run Model
        val inputs = arrayOfNulls<Any>(2)
        inputs[imageInputIndex] = inputImageBuffer.buffer
        inputs[maskInputIndex] = maskBuffer.buffer
        interpreter.runForMultipleInputsOutputs(inputs, mapOf(0 to outputBuffer.buffer))
        
        // 6. Seamless Recombination
        val rawInpainted512 = tensorBufferToBitmapSafely(outputBuffer, inputSize, outIsNCHW)
        
        val dilatedMask512 = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val dilatedPixelsInt = IntArray(inputSize * inputSize)
        for (i in dilatedMaskPixels.indices) dilatedPixelsInt[i] = if (dilatedMaskPixels[i]) Color.WHITE else Color.BLACK
        dilatedMask512.setPixels(dilatedPixelsInt, 0, inputSize, 0, 0, inputSize, inputSize)
        
        val upscaledInpainted = Bitmap.createScaledBitmap(rawInpainted512, cropSize, cropSize, true)
        val upscaledDilatedMask = Bitmap.createScaledBitmap(dilatedMask512, cropSize, cropSize, true)
        
        val finalImage = image.copy(Bitmap.Config.ARGB_8888, true)
        val finalCanvas = Canvas(finalImage)
        
        val patch = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        val resultPixels = IntArray(cropSize * cropSize)
        val maskPixelsToPaste = IntArray(cropSize * cropSize)
        upscaledInpainted.getPixels(resultPixels, 0, cropSize, 0, 0, cropSize, cropSize)
        upscaledDilatedMask.getPixels(maskPixelsToPaste, 0, cropSize, 0, 0, cropSize, cropSize)
        
        val patchPixels = IntArray(cropSize * cropSize)
        for (i in patchPixels.indices) {
            patchPixels[i] = if (((maskPixelsToPaste[i] shr 16) and 0xFF) > 128) resultPixels[i] else Color.TRANSPARENT
        }
        patch.setPixels(patchPixels, 0, cropSize, 0, 0, cropSize, cropSize)
        
        finalCanvas.drawBitmap(patch, -dx, -dy, null)
        
        imageCrop.recycle(); maskCrop.recycle(); resizedImg.recycle(); resizedMask.recycle()
        rawInpainted512.recycle(); dilatedMask512.recycle()
        upscaledInpainted.recycle(); upscaledDilatedMask.recycle(); patch.recycle()
        
        finalImage
    }

    private fun getMaskBounds(mask: Bitmap): Rect? {
        var minX = mask.width; var minY = mask.height; var maxX = 0; var maxY = 0
        val pixels = IntArray(mask.width * mask.height)
        mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)
        for (y in 0 until mask.height) {
            for (x in 0 until mask.width) {
                if (((pixels[y * mask.width + x] shr 16) and 0xFF) > 100) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        if (minX > maxX) return null
        val padX = (maxX - minX) / 10
        val padY = (maxY - minY) / 10
        return Rect(
            (minX - padX).coerceAtLeast(0),
            (minY - padY).coerceAtLeast(0),
            (maxX + padX).coerceAtMost(mask.width),
            (maxY + padY).coerceAtMost(mask.height)
        )
    }
    
    private fun tensorBufferToBitmapSafely(buffer: TensorBuffer, size: Int, isNCHW: Boolean): Bitmap {
        val output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val floats = buffer.floatArray
        
        // CRITICAL FIX: Scan the entire array to prevent the parser from being blinded by black padding
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE
        for (f in floats) {
            if (f > maxVal) maxVal = f
            if (f < minVal) minVal = f
        }
        
        val isMinusOneToOne = minVal < -0.1f
        val isZeroToOne = !isMinusOneToOne && maxVal <= 1.5f
        val pixels = IntArray(size * size)
        val area = size * size
        
        for (y in 0 until size) {
            for (x in 0 until size) {
                val i = y * size + x
                val rF = if (isNCHW) floats[0 * area + i] else floats[i * 3 + 0]
                val gF = if (isNCHW) floats[1 * area + i] else floats[i * 3 + 1]
                val bF = if (isNCHW) floats[2 * area + i] else floats[i * 3 + 2]
                
                val r = normalizeColor(rF, isMinusOneToOne, isZeroToOne)
                val g = normalizeColor(gF, isMinusOneToOne, isZeroToOne)
                val b = normalizeColor(bF, isMinusOneToOne, isZeroToOne)
                pixels[i] = Color.rgb(r, g, b)
            }
        }
        output.setPixels(pixels, 0, size, 0, 0, size, size)
        return output
    }
    
    private fun normalizeColor(value: Float, isMinusOneToOne: Boolean, isZeroToOne: Boolean): Int {
        val normalized = when {
            isMinusOneToOne -> (value + 1f) / 2f
            isZeroToOne -> value
            else -> value / 255f
        }
        return (normalized.coerceIn(0f, 1f) * 255).toInt()
    }
}
