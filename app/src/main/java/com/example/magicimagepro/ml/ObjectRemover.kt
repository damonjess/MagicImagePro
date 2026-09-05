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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class ObjectRemover(context: Context) : TFLiteModel(context, "lama_dilated-tflite-float.tflite") {

    private val modelWidth = 512
    private val modelHeight = 512

    suspend fun removeObject(image: Bitmap, mask: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        val safeImage = image.copy(image.config ?: Bitmap.Config.ARGB_8888, true)
        val safeMask = mask.copy(Bitmap.Config.ARGB_8888, true)

        val (dilatedImage, dilatedMask) = dilateMaskForCoverage(safeImage, safeMask)

        val paddedImage = padToSquareReflect(dilatedImage)
        val paddedMask = padToSquareBlack(dilatedMask)

        val scaledImage = Bitmap.createScaledBitmap(paddedImage, modelWidth, modelHeight, true)
        val scaledMask = Bitmap.createScaledBitmap(paddedMask, modelWidth, modelHeight, true)

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

        val rawAiOutput = convertOutputToBitmap(outputBuffer, modelWidth, modelHeight, outIsNCHW)

        val inpaintedPadded = Bitmap.createScaledBitmap(rawAiOutput, paddedImage.width, paddedImage.height, true)
        val inpaintedFull = cropFromSquare(inpaintedPadded, safeImage.width, safeImage.height)

        val colorCorrected = matchBorderColors(safeImage, inpaintedFull, dilatedMask)

        val maxDim = max(safeImage.width, safeImage.height).toFloat()
        val adaptiveBlur = (maxDim / 120f).coerceIn(8f, 32f)
        val widerBlur = (adaptiveBlur * 1.8f).coerceAtMost(50f)

        val softMaskOuter = blurMask(dilatedMask, blurRadius = widerBlur)
        val softMaskInner = blurMask(dilatedMask, blurRadius = adaptiveBlur * 0.5f)

        val gradientMask = buildGradientTransitionMask(softMaskOuter, softMaskInner)

        val seamlessComposite = seamlessPoissonBlend(safeImage, colorCorrected, gradientMask, dilatedMask)

        val finalSeamCleanup = cleanupSeamLine(safeImage, seamlessComposite, dilatedMask, adaptiveBlur)

        safeImage.recycle()
        safeMask.recycle()
        dilatedImage.recycle()
        dilatedMask.recycle()
        paddedImage.recycle()
        paddedMask.recycle()
        scaledImage.recycle()
        scaledMask.recycle()
        rawAiOutput.recycle()
        inpaintedPadded.recycle()
        inpaintedFull.recycle()
        colorCorrected.recycle()
        softMaskOuter.recycle()
        softMaskInner.recycle()
        gradientMask.recycle()
        seamlessComposite.recycle()

        finalSeamCleanup
    }

    // --- Helper Methods ---

    private fun dilateMaskForCoverage(image: Bitmap, mask: Bitmap): Pair<Bitmap, Bitmap> {
        val outImage = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        val outMask = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)

        val w = image.width
        val h = image.height
        val maxDim = max(w, h)
        val radiusPx = (maxDim / 220f).coerceIn(3f, 14f).toInt()

        val srcPixels = IntArray(w * h)
        mask.getPixels(srcPixels, 0, w, 0, 0, w, h)
        val maskPixels = IntArray(w * h)
        mask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val temp = IntArray(w * h)

        var passes = 0
        var r = radiusPx
        while (r > 0 && passes < 5) {
            val block = if (r >= 5) 5 else if (r >= 3) 3 else 1
            val half = block / 2

            for (y in 0 until h) {
                for (x in 0 until w) {
                    if (Color.red(maskPixels[y * w + x]) > 127) {
                        for (dy in -half..half) {
                            val ny = y + dy
                            if (ny in 0 until h) {
                                val rowStart = ny * w
                                val xStart = max(0, x - half)
                                val xEnd = min(w - 1, x + half)
                                for (nx in xStart..xEnd) {
                                    temp[rowStart + nx] = 0xFFFFFFFF.toInt()
                                }
                            }
                        }
                    }
                }
            }

            for (y in 0 until h) {
                for (x in 0 until w) {
                    if (temp[y * w + x] == 0xFFFFFFFF.toInt()) {
                        maskPixels[y * w + x] = 0xFFFFFFFF.toInt()
                    }
                    temp[y * w + x] = 0
                }
            }

            r -= block
            passes++
        }

        val edgePixels = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val m = Color.red(maskPixels[idx]) > 127
                var isEdge = false
                if (m) {
                    for (dy in -1..1) {
                        for (dx in -1..1) {
                            val nx = x + dx
                            val ny = y + dy
                            if (nx in 0 until w && ny in 0 until h) {
                                if (Color.red(maskPixels[ny * w + nx]) <= 127) {
                                    isEdge = true
                                    break
                                }
                            } else {
                                isEdge = true
                                break
                            }
                        }
                        if (isEdge) break
                    }
                    if (isEdge) edgePixels[idx] = 0xFFFFFFFF.toInt()
                }
            }
        }

        val edgeDilate = 2
        for (pass in 0 until edgeDilate) {
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val idx = y * w + x
                    if (edgePixels[idx] == 0xFFFFFFFF.toInt()) {
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                val nx = x + dx
                                val ny = y + dy
                                if (nx in 0 until w && ny in 0 until h) {
                                    temp[ny * w + nx] = 0xFFFFFFFF.toInt()
                                }
                            }
                        }
                    }
                }
            }
            for (i in temp.indices) {
                if (temp[i] == 0xFFFFFFFF.toInt()) {
                    maskPixels[i] = 0xFFFFFFFF.toInt()
                }
                temp[i] = 0
            }
        }

        val cvSrc = Canvas(outImage)
        cvSrc.drawBitmap(image, 0f, 0f, null)
        outMask.setPixels(maskPixels, 0, w, 0, 0, w, h)
        return Pair(outImage, outMask)
    }

    private fun padToSquareReflect(bitmap: Bitmap): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        if (bitmap.width == maxDim && bitmap.height == maxDim) {
            return bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
        }

        val padded = Bitmap.createBitmap(maxDim, maxDim, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)

        val left = (maxDim - bitmap.width) / 2
        val top = (maxDim - bitmap.height) / 2

        canvas.drawBitmap(bitmap, left.toFloat(), top.toFloat(), null)

        val srcPixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(srcPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val dstPixels = IntArray(maxDim * maxDim)
        padded.getPixels(dstPixels, 0, maxDim, 0, 0, maxDim, maxDim)

        for (y in 0 until maxDim) {
            for (x in 0 until maxDim) {
                val srcX: Int
                val srcY: Int

                val bx = x - left
                val by = y - top

                srcX = when {
                    bx < 0 -> -bx
                    bx >= bitmap.width -> 2 * (bitmap.width - 1) - bx
                    else -> bx
                }
                srcY = when {
                    by < 0 -> -by
                    by >= bitmap.height -> 2 * (bitmap.height - 1) - by
                    else -> by
                }

                val clampedX = srcX.coerceIn(0, bitmap.width - 1)
                val clampedY = srcY.coerceIn(0, bitmap.height - 1)

                dstPixels[y * maxDim + x] = srcPixels[clampedY * bitmap.width + clampedX]
            }
        }

        padded.setPixels(dstPixels, 0, maxDim, 0, 0, maxDim, maxDim)
        return padded
    }

    private fun padToSquareBlack(bitmap: Bitmap): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        if (bitmap.width == maxDim && bitmap.height == maxDim) {
            return bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
        }

        val padded = Bitmap.createBitmap(maxDim, maxDim, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawColor(Color.BLACK)

        val left = (maxDim - bitmap.width) / 2f
        val top = (maxDim - bitmap.height) / 2f
        canvas.drawBitmap(bitmap, left, top, null)

        return padded
    }

    private fun cropFromSquare(squaredBitmap: Bitmap, originalWidth: Int, originalHeight: Int): Bitmap {
        if (squaredBitmap.width == originalWidth && squaredBitmap.height == originalHeight) {
            return squaredBitmap.copy(squaredBitmap.config ?: Bitmap.Config.ARGB_8888, true)
        }

        val left = (squaredBitmap.width - originalWidth) / 2
        val top = (squaredBitmap.height - originalHeight) / 2

        return Bitmap.createBitmap(squaredBitmap, left, top, originalWidth, originalHeight)
    }

    private fun buildGradientTransitionMask(outerMask: Bitmap, innerMask: Bitmap): Bitmap {
        val w = outerMask.width
        val h = outerMask.height
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        val outerPx = IntArray(w * h)
        val innerPx = IntArray(w * h)
        outerMask.getPixels(outerPx, 0, w, 0, 0, w, h)
        innerMask.getPixels(innerPx, 0, w, 0, 0, w, h)

        val outPx = IntArray(w * h)
        for (i in 0 until w * h) {
            val o = Color.red(outerPx[i]).toFloat() / 255f
            val inn = Color.red(innerPx[i]).toFloat() / 255f
            val blend = (o + inn) * 0.5f
            val a = (blend.coerceIn(0f, 1f) * 255).toInt()
            outPx[i] = Color.argb(a, 255, 255, 255)
        }
        result.setPixels(outPx, 0, w, 0, 0, w, h)
        return result
    }

    private fun matchBorderColors(original: Bitmap, inpainted: Bitmap, mask: Bitmap): Bitmap {
        val w = original.width
        val h = original.height

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val origPx = IntArray(w * h)
        val inPx = IntArray(w * h)
        val maskPx = IntArray(w * h)

        original.getPixels(origPx, 0, w, 0, 0, w, h)
        inpainted.getPixels(inPx, 0, w, 0, 0, w, h)
        mask.getPixels(maskPx, 0, w, 0, 0, w, h)

        val bandWidth = (min(w, h) / 50f).coerceIn(4f, 16f).toInt()

        data class RGB(val r: Long, val g: Long, val b: Long, val count: Long)

        fun sampleRing(useOrigSide: Boolean): RGB {
            var sr = 0L
            var sg = 0L
            var sb = 0L
            var count = 0L

            for (y in 0 until h) {
                for (x in 0 until w) {
                    val idx = y * w + x
                    val mv = Color.red(maskPx[idx])

                    val inBand = if (useOrigSide) {
                        mv in 1..127
                    } else {
                        mv in 128..254
                    }

                    if (inBand) {
                        val pxArr = if (useOrigSide) origPx else inPx
                        var nearEdge = false
                        for (dy in -bandWidth..bandWidth) {
                            for (dx in -bandWidth..bandWidth) {
                                val nx = x + dx
                                val ny = y + dy
                                if (nx in 0 until w && ny in 0 until h) {
                                    val nmv = Color.red(maskPx[ny * w + nx])
                                    if ((useOrigSide && nmv > 127) || (!useOrigSide && nmv <= 127)) {
                                        nearEdge = true
                                        break
                                    }
                                }
                            }
                            if (nearEdge) break
                        }
                        if (nearEdge) {
                            val p = pxArr[idx]
                            sr += Color.red(p)
                            sg += Color.green(p)
                            sb += Color.blue(p)
                            count++
                        }
                    }
                }
            }
            return RGB(sr, sg, sb, count)
        }

        val origSample = sampleRing(useOrigSide = true)
        val inpSample = sampleRing(useOrigSide = false)

        val outPx = IntArray(w * h)
        System.arraycopy(inPx, 0, outPx, 0, w * h)

        if (origSample.count > 50 && inpSample.count > 50) {
            val oR = origSample.r.toFloat() / origSample.count
            val oG = origSample.g.toFloat() / origSample.count
            val oB = origSample.b.toFloat() / origSample.count

            val iR = inpSample.r.toFloat() / inpSample.count
            val iG = inpSample.g.toFloat() / inpSample.count
            val iB = inpSample.b.toFloat() / inpSample.count

            val dR = oR - iR
            val dG = oG - iG
            val dB = oB - iB

            val rScale = if (iR > 1f) (oR / iR) else 1f
            val gScale = if (iG > 1f) (oG / iG) else 1f
            val bScale = if (iB > 1f) (oB / iB) else 1f

            val useScale = maxOf(rScale, gScale, bScale, 1f / rScale, 1f / gScale, 1f / bScale) < 1.25f

            for (y in 0 until h) {
                for (x in 0 until w) {
                    val idx = y * w + x
                    if (Color.red(maskPx[idx]) > 0) {
                        val p = inPx[idx]
                        var r = Color.red(p).toFloat()
                        var g = Color.green(p).toFloat()
                        var b = Color.blue(p).toFloat()

                        if (useScale) {
                            r *= rScale
                            g *= gScale
                            b *= bScale
                        } else {
                            r += dR
                            g += dG
                            b += dB
                        }

                        val borderDist = distanceToMaskBorder(x, y, maskPx, w, h)
                        val t = (borderDist / 40f).coerceIn(0f, 1f)
                        val mixR = r * t + (Color.red(p) + dR) * (1f - t)
                        val mixG = g * t + (Color.green(p) + dG) * (1f - t)
                        val mixB = b * t + (Color.blue(p) + dB) * (1f - t)

                        val rr = mixR.coerceIn(0f, 255f).toInt()
                        val gg = mixG.coerceIn(0f, 255f).toInt()
                        val bb = mixB.coerceIn(0f, 255f).toInt()

                        outPx[idx] = Color.rgb(rr, gg, bb)
                    }
                }
            }
        }

        result.setPixels(outPx, 0, w, 0, 0, w, h)
        return result
    }

    private fun distanceToMaskBorder(x: Int, y: Int, maskPx: IntArray, w: Int, h: Int): Float {
        val centerInside = Color.red(maskPx[y * w + x]) > 127
        val maxSearch = 60
        for (d in 1..maxSearch) {
            for (dy in -d..d) {
                val ny = y + dy
                if (ny !in 0 until h) continue
                val dx1 = -d
                val dx2 = d
                for (dx in listOf(dx1, dx2)) {
                    val nx = x + dx
                    if (nx in 0 until w) {
                        val v = Color.red(maskPx[ny * w + nx]) > 127
                        if (v != centerInside) return d.toFloat()
                    }
                }
            }
            for (dx in -d + 1 until d) {
                val nx = x + dx
                if (nx !in 0 until w) continue
                val dy1 = -d
                val dy2 = d
                for (dy in listOf(dy1, dy2)) {
                    val ny = y + dy
                    if (ny in 0 until h) {
                        val v = Color.red(maskPx[ny * w + nx]) > 127
                        if (v != centerInside) return d.toFloat()
                    }
                }
            }
        }
        return maxSearch.toFloat()
    }

    private fun seamlessPoissonBlend(original: Bitmap, inpainted: Bitmap, blendMask: Bitmap, hardMask: Bitmap): Bitmap {
        val w = original.width
        val h = original.height

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawBitmap(original, 0f, 0f, null)

        val blendBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val patchCanvas = Canvas(blendBitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        patchCanvas.drawBitmap(blendMask, 0f, 0f, null)
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
        patchCanvas.drawBitmap(inpainted, 0f, 0f, paint)

        canvas.drawBitmap(blendBitmap, 0f, 0f, null)

        blendBitmap.recycle()
        return result
    }

    private fun cleanupSeamLine(original: Bitmap, composited: Bitmap, mask: Bitmap, blurAmount: Float): Bitmap {
        val w = original.width
        val h = original.height

        val seamWidth = (blurAmount * 0.6f).coerceIn(3f, 18f)

        val seamMaskPixels = IntArray(w * h)
        val maskPx = IntArray(w * h)
        mask.getPixels(maskPx, 0, w, 0, 0, w, h)

        val maxDim = max(w, h)
        val searchR = (maxDim / 80f).coerceIn(2f, 12f).toInt()

        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val centerInside = Color.red(maskPx[idx]) > 127
                var onBorder = false

                val half = searchR
                for (dy in -half..half) {
                    for (dx in -half..half) {
                        val nx = x + dx
                        val ny = y + dy
                        if (nx in 0 until w && ny in 0 until h) {
                            val v = Color.red(maskPx[ny * w + nx]) > 127
                            if (v != centerInside) {
                                onBorder = true
                                break
                            }
                        }
                    }
                    if (onBorder) break
                }

                if (onBorder) {
                    var insideDist = Int.MAX_VALUE
                    var outsideDist = Int.MAX_VALUE
                    for (d in 1..(seamWidth * 2.5f).toInt()) {
                        if (insideDist == Int.MAX_VALUE) {
                            searchBorder@ for (dy in -d..d) {
                                for (dx in -d..d) {
                                    if (kotlin.math.abs(dx) != d && kotlin.math.abs(dy) != d) continue
                                    val nx = x + dx
                                    val ny = y + dy
                                    if (nx in 0 until w && ny in 0 until h) {
                                        if (Color.red(maskPx[ny * w + nx]) > 127 == centerInside) {
                                            insideDist = d
                                            break@searchBorder
                                        }
                                    }
                                }
                            }
                        }
                        if (outsideDist == Int.MAX_VALUE) {
                            searchOuter@ for (dy in -d..d) {
                                for (dx in -d..d) {
                                    if (kotlin.math.abs(dx) != d && kotlin.math.abs(dy) != d) continue
                                    val nx = x + dx
                                    val ny = y + dy
                                    if (nx in 0 until w && ny in 0 until h) {
                                        if (Color.red(maskPx[ny * w + nx]) > 127 != centerInside) {
                                            outsideDist = d
                                            break@searchOuter
                                        }
                                    }
                                }
                            }
                        }
                        if (insideDist != Int.MAX_VALUE && outsideDist != Int.MAX_VALUE) break
                    }
                    val id = if (insideDist == Int.MAX_VALUE) seamWidth.toInt() else insideDist
                    val od = if (outsideDist == Int.MAX_VALUE) seamWidth.toInt() else outsideDist
                    val minDist = min(id, od)
                    if (minDist <= seamWidth) {
                        val falloff = 1f - (minDist.toFloat() / seamWidth)
                        val a = (falloff * 255f).coerceIn(0f, 255f).toInt()
                        seamMaskPixels[idx] = Color.argb(a, 255, 255, 255)
                    }
                }
            }
        }

        val seamMaskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        seamMaskBitmap.setPixels(seamMaskPixels, 0, w, 0, 0, w, h)

        val blurredSeamMask = blurMaskGeneric(seamMaskBitmap, (seamWidth * 0.7f).coerceIn(3f, 20f))

        val origPx = IntArray(w * h)
        val compPx = IntArray(w * h)
        val bsmPx = IntArray(w * h)
        original.getPixels(origPx, 0, w, 0, 0, w, h)
        composited.getPixels(compPx, 0, w, 0, 0, w, h)
        blurredSeamMask.getPixels(bsmPx, 0, w, 0, 0, w, h)

        val averagedPx = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val seamAlpha = Color.alpha(bsmPx[idx]).toFloat() / 255f

                if (seamAlpha <= 0.01f) {
                    averagedPx[idx] = compPx[idx]
                    continue
                }

                val r = 3
                var sr = 0f
                var sg = 0f
                var sb = 0f
                var wt = 0f
                for (dy in -r..r) {
                    for (dx in -r..r) {
                        val nx = x + dx
                        val ny = y + dy
                        if (nx in 0 until w && ny in 0 until h) {
                            val d = sqrt((dx * dx + dy * dy).toFloat())
                            val wd = 1f / (1f + d)
                            val p = compPx[ny * w + nx]
                            sr += Color.red(p) * wd
                            sg += Color.green(p) * wd
                            sb += Color.blue(p) * wd
                            wt += wd
                        }
                    }
                }
                val avgR = sr / wt
                val avgG = sg / wt
                val avgB = sb / wt

                val cp = compPx[idx]
                val cr = Color.red(cp).toFloat()
                val cg = Color.green(cp).toFloat()
                val cb = Color.blue(cp).toFloat()

                val mr = cr * (1f - seamAlpha) + avgR * seamAlpha
                val mg = cg * (1f - seamAlpha) + avgG * seamAlpha
                val mb = cb * (1f - seamAlpha) + avgB * seamAlpha

                averagedPx[idx] = Color.rgb(
                    mr.coerceIn(0f, 255f).toInt(),
                    mg.coerceIn(0f, 255f).toInt(),
                    mb.coerceIn(0f, 255f).toInt()
                )
            }
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(averagedPx, 0, w, 0, 0, w, h)

        seamMaskBitmap.recycle()
        blurredSeamMask.recycle()
        return result
    }

    private fun blurMask(mask: Bitmap, blurRadius: Float): Bitmap {
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
        paint.maskFilter = BlurMaskFilter(blurRadius.coerceIn(1f, 25f), BlurMaskFilter.Blur.NORMAL)

        val alphaMask = alphaSource.extractAlpha(paint, null)

        val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG)
        drawPaint.color = Color.WHITE
        canvas.drawBitmap(alphaMask, 0f, 0f, drawPaint)

        alphaMask.recycle()
        alphaSource.recycle()
        return outputBitmap
    }

    private fun blurMaskGeneric(mask: Bitmap, blurRadius: Float): Bitmap {
        return blurMask(mask, blurRadius)
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
