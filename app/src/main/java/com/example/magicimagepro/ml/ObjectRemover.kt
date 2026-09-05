package com.example.magicimagepro.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class ObjectRemover(context: Context) : TFLiteModel(context, "lama_dilated-tflite-float.tflite") {

    private val modelWidth = 512
    private val modelHeight = 512

    suspend fun removeObject(image: Bitmap, mask: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        val w = image.width
        val h = image.height

        val safeImage = image.copy(image.config ?: Bitmap.Config.ARGB_8888, true)
        val safeMask = mask.copy(Bitmap.Config.ARGB_8888, true)

        val (dilatedImage, dilatedMask) = dilateMaskForCoverage(safeImage, safeMask)
        safeImage.recycle()
        safeMask.recycle()

        val maskPx = IntArray(w * h)
        dilatedMask.getPixels(maskPx, 0, w, 0, 0, w, h)

        val dist = bfsDistanceToUnmasked(maskPx, w, h)
        val roi = computeMaskRoi(maskPx, w, h, contextPadFrac = 0.55f)

        val roiW = roi.right - roi.left
        val roiH = roi.bottom - roi.top
        val roiArea = roiW * roiH
        val fullArea = w * h
        val useRoi = roiArea < fullArea * 0.25f && roiW > 0 && roiH > 0

        val aiInpainted: Bitmap = if (useRoi) {
            val cropImg = Bitmap.createBitmap(dilatedImage, roi.left, roi.top, roiW, roiH)
            val cropMask = Bitmap.createBitmap(dilatedMask, roi.left, roi.top, roiW, roiH)
            val inpaintedCrop = runLaMaOnBitmap(cropImg, cropMask)
            cropImg.recycle()
            cropMask.recycle()

            val fullResult = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val c = Canvas(fullResult)
            c.drawBitmap(dilatedImage, 0f, 0f, null)
            c.drawBitmap(inpaintedCrop, roi.left.toFloat(), roi.top.toFloat(), null)
            inpaintedCrop.recycle()
            fullResult
        } else {
            runLaMaOnBitmap(dilatedImage, dilatedMask)
        }

        val colorCorrected = matchBorderColors(dilatedImage, aiInpainted, maskPx, dist, w, h)
        aiInpainted.recycle()

        val maxDim = max(w, h).toFloat()
        val adaptiveBlur = (maxDim / 110f).coerceIn(10f, 42f)
        val widerBlur = (adaptiveBlur * 2.2f).coerceAtMost(65f)

        val softMaskOuter = blurMask(dilatedMask, blurRadius = widerBlur)
        val softMaskInner = blurMask(dilatedMask, blurRadius = adaptiveBlur * 0.45f)
        val gradientMask = buildGradientTransitionMask(softMaskOuter, softMaskInner)

        val seamless = seamlessComposite(dilatedImage, colorCorrected, gradientMask)
        colorCorrected.recycle()
        softMaskOuter.recycle()
        softMaskInner.recycle()
        gradientMask.recycle()

        val final = cleanupSeamLine(seamless, maskPx, dist, adaptiveBlur, w, h)
        seamless.recycle()
        dilatedImage.recycle()
        dilatedMask.recycle()

        final
    }

    // ---------------------------------------------------------------------
    // Fast distance transform (BFS, O(w*h) total)
    // ---------------------------------------------------------------------

    private fun bfsDistanceToUnmasked(maskPx: IntArray, w: Int, h: Int): IntArray {
        val total = w * h
        val dist = IntArray(total) { -1 }
        val q = ArrayDeque<Int>(total / 8)
        for (i in 0 until total) {
            if (!isMaskPixel(maskPx[i])) {
                dist[i] = 0
                q.addLast(i)
            }
        }
        val stepX = intArrayOf(1, -1, 0, 0)
        val stepY = intArrayOf(0, 0, 1, -1)
        while (q.isNotEmpty()) {
            val head = q.removeFirst()
            val x = head % w
            val y = head / w
            val d = dist[head]
            for (k in 0..3) {
                val nx = x + stepX[k]
                val ny = y + stepY[k]
                if (nx in 0 until w && ny in 0 until h) {
                    val ni = ny * w + nx
                    if (dist[ni] == -1) {
                        dist[ni] = d + 1
                        q.addLast(ni)
                    }
                }
            }
        }
        return dist
    }

    // ---------------------------------------------------------------------
    // Model runner
    // ---------------------------------------------------------------------

    private fun isMaskPixel(c: Int): Boolean {
        return Color.red(c) > 127 || Color.green(c) > 127 || Color.alpha(c) > 127
    }

    private fun runLaMaOnBitmap(input: Bitmap, inputMask: Bitmap): Bitmap {
        val w = input.width
        val h = input.height

        val (squareImg, squareMask, padL, padT, padR, padB, squareSize) =
            padToSquareReflect(input, inputMask)

        val scaledImg = Bitmap.createScaledBitmap(squareImg, modelWidth, modelHeight, true)
        val scaledMask = Bitmap.createScaledBitmap(squareMask, modelWidth, modelHeight, true)
        squareImg.recycle()
        squareMask.recycle()

        val imgBuf = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * 3).order(ByteOrder.nativeOrder())
        val maskBuf = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * 1).order(ByteOrder.nativeOrder())
        val outputBuf = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * 3).order(ByteOrder.nativeOrder())

        val inputCount = interpreter.inputTensorCount
        require(inputCount == 2) { "LaMa model must have image and mask inputs" }
        val imageInputIndex = (0 until inputCount).firstOrNull { index ->
            val name = interpreter.getInputTensor(index).name().lowercase()
            name.contains("painted") || name.contains("image")
        } ?: 0
        val maskInputIndex = (0 until inputCount).firstOrNull { index ->
            interpreter.getInputTensor(index).name().lowercase().contains("mask")
        } ?: (1 - imageInputIndex)
        require(imageInputIndex != maskInputIndex) { "LaMa image and mask inputs are ambiguous" }

        val imageShape = interpreter.getInputTensor(imageInputIndex).shape()
        val maskShape = interpreter.getInputTensor(maskInputIndex).shape()
        val imageIsNchw = imageShape.size == 4 && imageShape[1] == 3
        val maskIsNchw = maskShape.size == 4 && maskShape[1] == 1

        fillImageBuffer(scaledImg, imgBuf, imageIsNchw)
        fillMaskBuffer(scaledMask, maskBuf, maskIsNchw)
        scaledImg.recycle()
        scaledMask.recycle()

        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(imgBuf, maskBuf).also { inputs ->
                inputs[imageInputIndex] = imgBuf
                inputs[maskInputIndex] = maskBuf
            },
            mapOf(0 to outputBuf)
        )

        val outputShape = interpreter.getOutputTensor(0).shape()
        val isNchwOutput = outputShape != null && outputShape.size == 4 && outputShape[1] == 3

        val rawSquare = convertOutputToBitmap(outputBuf, modelWidth, modelHeight, isNchwOutput)

        val cropL = (padL * modelWidth / squareSize).coerceIn(0, modelWidth - 1)
        val cropT = (padT * modelHeight / squareSize).coerceIn(0, modelHeight - 1)
        val cropR = (modelWidth - padR * modelWidth / squareSize).coerceIn(cropL + 1, modelWidth)
        val cropB = (modelHeight - padB * modelHeight / squareSize).coerceIn(cropT + 1, modelHeight)
        val cropped = if (cropL == 0 && cropT == 0 && cropR == modelWidth && cropB == modelHeight) {
            rawSquare
        } else {
            val cw = cropR - cropL
            val ch = cropB - cropT
            if (cw <= 0 || ch <= 0) {
                rawSquare
            } else {
                Bitmap.createBitmap(rawSquare, cropL, cropT, cw, ch).also { rawSquare.recycle() }
            }
        }
        val resized = if (cropped.width == w && cropped.height == h) cropped
        else Bitmap.createScaledBitmap(cropped, w, h, true).also { cropped.recycle() }
        return resized
    }

    private fun fillImageBuffer(img: Bitmap, imgBuf: ByteBuffer, isNchw: Boolean) {
        val w = img.width
        val h = img.height
        val total = w * h
        val iPx = IntArray(total)
        img.getPixels(iPx, 0, w, 0, 0, w, h)
        imgBuf.rewind()

        if (isNchw) {
            for (i in 0 until total) {
                imgBuf.putFloat(Color.red(iPx[i]) / 255f)
            }
            for (i in 0 until total) {
                imgBuf.putFloat(Color.green(iPx[i]) / 255f)
            }
            for (i in 0 until total) {
                imgBuf.putFloat(Color.blue(iPx[i]) / 255f)
            }
        } else {
            for (i in 0 until total) {
                val c = iPx[i]
                imgBuf.putFloat(Color.red(c) / 255f)
                imgBuf.putFloat(Color.green(c) / 255f)
                imgBuf.putFloat(Color.blue(c) / 255f)

            }
        }
    }

    private fun fillMaskBuffer(mask: Bitmap, maskBuf: ByteBuffer, isNchw: Boolean) {
        val w = mask.width
        val h = mask.height
        val total = w * h
        val mPx = IntArray(total)
        mask.getPixels(mPx, 0, w, 0, 0, w, h)
        maskBuf.rewind()

        for (i in 0 until total) {
            maskBuf.putFloat(if (isMaskPixel(mPx[i])) 1f else 0f)
        }
    }

    private fun convertOutputToBitmap(buf: ByteBuffer, w: Int, h: Int, isNchw: Boolean): Bitmap {
        buf.rewind()
        val total = w * h
        val out = IntArray(total)

        if (isNchw) {
            val rOffset = 0
            val gOffset = total * 4
            val bOffset = total * 8
            for (i in 0 until total) {
                val r = (buf.getFloat(rOffset + i * 4) * 255f).coerceIn(0f, 255f).toInt()
                val g = (buf.getFloat(gOffset + i * 4) * 255f).coerceIn(0f, 255f).toInt()
                val b = (buf.getFloat(bOffset + i * 4) * 255f).coerceIn(0f, 255f).toInt()
                out[i] = Color.rgb(r, g, b)
            }
        } else {
            for (i in 0 until total) {
                val r = (buf.float * 255f).coerceIn(0f, 255f).toInt()
                val g = (buf.float * 255f).coerceIn(0f, 255f).toInt()
                val b = (buf.float * 255f).coerceIn(0f, 255f).toInt()
                out[i] = Color.rgb(r, g, b)
            }
        }
        val bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bm.setPixels(out, 0, w, 0, 0, w, h)
        return bm
    }

    // ---------------------------------------------------------------------
    // ROI crop helpers
    // ---------------------------------------------------------------------

    private fun computeMaskRoi(
        maskPx: IntArray, w: Int, h: Int, contextPadFrac: Float
    ): Rect {
        var minX = w
        var minY = h
        var maxX = -1
        var maxY = -1
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (isMaskPixel(maskPx[y * w + x])) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        if (maxX < 0 || maxY < 0) {
            return Rect(0, 0, w, h)
        }
        val bw = maxX - minX + 1
        val bh = maxY - minY + 1
        val pad = (max(bw, bh) * contextPadFrac).roundToInt()

        var l = (minX - pad).coerceIn(0, w - 1)
        var t = (minY - pad).coerceIn(0, h - 1)
        var r = (maxX + pad).coerceIn(1, w)
        var b = (maxY + pad).coerceIn(1, h)

        val rw = r - l
        val rh = b - t
        val side = max(rw, rh)
        if (side > rw) {
            val extraL = (side - rw) / 2
            val extraR = side - rw - extraL
            l = (l - extraL).coerceIn(0, w - 1)
            r = (r + extraR).coerceIn(1, w)
            if (r - l < side) {
                val shift = side - (r - l)
                l = (l - shift).coerceAtLeast(0)
            } else if (r - l > side) {
                r = l + side
            }
        }
        if (side > rh) {
            val extraT = (side - rh) / 2
            val extraB = side - rh - extraT
            t = (t - extraT).coerceIn(0, h - 1)
            b = (b + extraB).coerceIn(1, h)
            if (b - t < side) {
                val shift = side - (b - t)
                t = (t - shift).coerceAtLeast(0)
            } else if (b - t > side) {
                b = t + side
            }
        }

        if (l < 0) l = 0
        if (t < 0) t = 0
        if (r > w) r = w
        if (b > h) b = h
        return Rect(l, t, r, b)
    }

    // ---------------------------------------------------------------------
    // Mask dilation for coverage (fast block-dilate, limited rounds)
    // ---------------------------------------------------------------------

    private fun dilateMaskForCoverage(image: Bitmap, mask: Bitmap): Pair<Bitmap, Bitmap> {
        val w = image.width
        val h = image.height
        val maxDim = max(w, h)
        val rounds = when {
            maxDim <= 800 -> 2
            maxDim <= 1600 -> 3
            else -> 4
        }

        var workMask = mask.copy(Bitmap.Config.ARGB_8888, true)
        val wmPx = IntArray(w * h)
        val out = IntArray(w * h)
        for (round in 0 until rounds) {
            workMask.getPixels(wmPx, 0, w, 0, 0, w, h)
            System.arraycopy(wmPx, 0, out, 0, w * h)
            for (y in 1 until h - 1) {
                val row = y * w
                for (x in 1 until w - 1) {
                    val idx = row + x
                    if (isMaskPixel(wmPx[idx])) continue
                    val has =
                        isMaskPixel(wmPx[idx - w]) ||
                        isMaskPixel(wmPx[idx + w]) ||
                        isMaskPixel(wmPx[idx - 1]) ||
                        isMaskPixel(wmPx[idx + 1])
                    if (has) out[idx] = Color.WHITE
                }
            }
            workMask.setPixels(out, 0, w, 0, 0, w, h)
        }

        val dilMaskPx = IntArray(w * h)
        workMask.getPixels(dilMaskPx, 0, w, 0, 0, w, h)

        val median = medianBorderColorFast(image, dilMaskPx, w, h)
        val origPx = IntArray(w * h)
        image.getPixels(origPx, 0, w, 0, 0, w, h)
        val filledPx = IntArray(w * h)
        for (i in 0 until w * h) {
            filledPx[i] = if (isMaskPixel(dilMaskPx[i])) median else origPx[i]
        }
        val outImg = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        outImg.setPixels(filledPx, 0, w, 0, 0, w, h)

        return outImg to workMask
    }

    private fun medianBorderColorFast(image: Bitmap, maskPx: IntArray, w: Int, h: Int): Int {
        val orig = IntArray(w * h)
        image.getPixels(orig, 0, w, 0, 0, w, h)
        val rs = IntArray(4096)
        val gs = IntArray(4096)
        val bs = IntArray(4096)
        var n = 0
        for (y in 0 until h step 2) {
            val row = y * w
            for (x in 0 until w step 2) {
                val idx = row + x
                if (!isMaskPixel(maskPx[idx])) {
                    var adj = false
                    if (x > 0 && isMaskPixel(maskPx[idx - 1])) adj = true
                    else if (x < w - 1 && isMaskPixel(maskPx[idx + 1])) adj = true
                    else if (y > 0 && isMaskPixel(maskPx[idx - w])) adj = true
                    else if (y < h - 1 && isMaskPixel(maskPx[idx + w])) adj = true
                    if (adj && n < 4096) {
                        val c = orig[idx]
                        rs[n] = Color.red(c)
                        gs[n] = Color.green(c)
                        bs[n] = Color.blue(c)
                        n++
                    }
                }
            }
        }
        if (n == 0) return Color.rgb(128, 128, 128)
        rs.sort(0, n); gs.sort(0, n); bs.sort(0, n)
        val m = n / 2
        return Color.rgb(rs[m], gs[m], bs[m])
    }

    // ---------------------------------------------------------------------
    // Reflection padding to square
    // ---------------------------------------------------------------------

    private fun padToSquareReflect(
        img: Bitmap, mask: Bitmap
    ): Tuple7<Bitmap, Bitmap, Int, Int, Int, Int, Int> {
        val w = img.width
        val h = img.height
        val side = max(w, h)
        val padL = (side - w) / 2
        val padR = side - w - padL
        val padT = (side - h) / 2
        val padB = side - h - padT

        val imgPx = IntArray(w * h); img.getPixels(imgPx, 0, w, 0, 0, w, h)
        val maskPx = IntArray(w * h); mask.getPixels(maskPx, 0, w, 0, 0, w, h)
        val sqImgPx = IntArray(side * side)
        val sqMaskPx = IntArray(side * side)

        fun refl(i: Int, len: Int): Int {
            if (len == 1) return 0
            var v = i
            if (v < 0) v = -v
            val period = 2 * (len - 1)
            v %= period
            if (v >= len) v = period - v
            return v
        }
        for (sy in 0 until side) {
            val oy = refl(sy - padT, h)
            val rowO = oy * w
            val rowS = sy * side
            for (sx in 0 until side) {
                val ox = refl(sx - padL, w)
                sqImgPx[rowS + sx] = imgPx[rowO + ox]
                sqMaskPx[rowS + sx] = maskPx[rowO + ox]
            }
        }
        val sqImg = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        val sqMask = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        sqImg.setPixels(sqImgPx, 0, side, 0, 0, side, side)
        sqMask.setPixels(sqMaskPx, 0, side, 0, 0, side, side)
        return Tuple7(sqImg, sqMask, padL, padT, padR, padB, side)
    }

    private data class Tuple7<A, B, C, D, E, F, G>(
        val a: A, val b: B, val c: C, val d: D, val e: E, val f: F, val g: G
    )

    // ---------------------------------------------------------------------
    // Border color matching (uses precomputed BFS distance map)
    // ---------------------------------------------------------------------

    private fun matchBorderColors(
        original: Bitmap, inpainted: Bitmap,
        maskPx: IntArray, dist: IntArray, w: Int, h: Int
    ): Bitmap {
        val origPx = IntArray(w * h); original.getPixels(origPx, 0, w, 0, 0, w, h)
        val inPx = IntArray(w * h); inpainted.getPixels(inPx, 0, w, 0, 0, w, h)

        val band = 5
        var oR = 0f; var oG = 0f; var oB = 0f; var oN = 0
        var iR = 0f; var iG = 0f; var iB = 0f; var iN = 0
        for (i in 0 until w * h) {
            val d = dist[i]
            val isMasked = isMaskPixel(maskPx[i])
            if (isMasked && d in 1..band) {
                val c = inPx[i]
                iR += Color.red(c); iG += Color.green(c); iB += Color.blue(c); iN++
            } else if (!isMasked && d <= band) {
                val c = origPx[i]
                oR += Color.red(c); oG += Color.green(c); oB += Color.blue(c); oN++
            }
        }
        if (oN == 0 || iN == 0) {
            val r = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            r.setPixels(inPx, 0, w, 0, 0, w, h)
            return r
        }
        oR /= oN; oG /= oN; oB /= oN
        iR /= iN; iG /= iN; iB /= iN
        val gainR = if (iR > 1f) oR / iR else 1f
        val gainG = if (iG > 1f) oG / iG else 1f
        val gainB = if (iB > 1f) oB / iB else 1f
        val offR = oR - iR
        val offG = oG - iG
        val offB = oB - iB
        val useScale = (gainR in 0.6f..1.25f) && (gainG in 0.6f..1.25f) && (gainB in 0.6f..1.25f)

        val outPx = IntArray(w * h)
        val maxD = 40f
        for (i in 0 until w * h) {
            val isMasked = isMaskPixel(maskPx[i])
            if (!isMasked) {
                outPx[i] = origPx[i]
                continue
            }
            val d = dist[i].toFloat().coerceAtLeast(0f)
            val t = (d / maxD).coerceIn(0f, 1f)
            val mix = 1f - t
            val c = inPx[i]
            var r = Color.red(c).toFloat()
            var g = Color.green(c).toFloat()
            var b = Color.blue(c).toFloat()
            if (useScale) {
                r *= 1f + (gainR - 1f) * mix
                g *= 1f + (gainG - 1f) * mix
                b *= 1f + (gainB - 1f) * mix
            } else {
                r += offR * mix
                g += offG * mix
                b += offB * mix
            }
            outPx[i] = Color.rgb(r.coerceIn(0f, 255f).toInt(), g.coerceIn(0f, 255f).toInt(), b.coerceIn(0f, 255f).toInt())
        }
        val res = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        res.setPixels(outPx, 0, w, 0, 0, w, h)
        return res
    }

    // ---------------------------------------------------------------------
    // Mask blur + gradient transition
    // ---------------------------------------------------------------------

    private fun blurMask(mask: Bitmap, blurRadius: Float): Bitmap {
        val w = mask.width; val h = mask.height
        val radius = blurRadius.coerceAtLeast(1f)
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(result)
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            maskFilter = BlurMaskFilter(radius, BlurMaskFilter.Blur.NORMAL)
        }
        c.drawBitmap(mask, 0f, 0f, p)
        return result
    }

    private fun buildGradientTransitionMask(outerBlur: Bitmap, innerBlur: Bitmap): Bitmap {
        val w = outerBlur.width; val h = outerBlur.height
        val oPx = IntArray(w * h); outerBlur.getPixels(oPx, 0, w, 0, 0, w, h)
        val iPx = IntArray(w * h); innerBlur.getPixels(iPx, 0, w, 0, 0, w, h)
        val outPx = IntArray(w * h)
        val wInner = 0.35f
        val wOuter = 0.65f
        for (i in 0 until w * h) {
            val iv = Color.red(iPx[i]) / 255f
            val ov = Color.red(oPx[i]) / 255f
            val v = ((iv * wInner + ov * wOuter) * 255f).coerceIn(0f, 255f).toInt()
            outPx[i] = Color.argb(v, 255, 255, 255)
        }
        val r = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        r.setPixels(outPx, 0, w, 0, 0, w, h)
        return r
    }

    // ---------------------------------------------------------------------
    // Seamless composite (SRC_IN for alpha-masked inpaint, then SRC_OVER)
    // ---------------------------------------------------------------------

    private fun seamlessComposite(
        original: Bitmap, inpainted: Bitmap, alphaMask: Bitmap
    ): Bitmap {
        val w = original.width; val h = original.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(out)
        c.drawBitmap(original, 0f, 0f, null)

        val maskedInpaint = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val ci = Canvas(maskedInpaint)
        ci.drawBitmap(inpainted, 0f, 0f, null)
        val srcIn = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
        }
        ci.drawBitmap(alphaMask, 0f, 0f, srcIn)
        c.drawBitmap(maskedInpaint, 0f, 0f, null)
        maskedInpaint.recycle()
        return out
    }

    // ---------------------------------------------------------------------
    // Seam cleanup (small box blur weighted by precomputed dist to contour)
    // ---------------------------------------------------------------------

    private fun cleanupSeamLine(
        composited: Bitmap, maskPx: IntArray, dist: IntArray,
        blurRadius: Float, w: Int, h: Int
    ): Bitmap {
        val seamWidth = (blurRadius * 0.35f).coerceIn(2f, 6f).toInt()
        val compPx = IntArray(w * h); composited.getPixels(compPx, 0, w, 0, 0, w, h)
        val outPx = IntArray(w * h)
        System.arraycopy(compPx, 0, outPx, 0, w * h)

        val r = seamWidth
        for (y in r until h - r) {
            val row = y * w
            for (x in r until w - r) {
                val idx = row + x
                val d = dist[idx]
                val falloff: Float
                if (d <= seamWidth) {
                    falloff = 1f - (d.toFloat() / seamWidth)
                } else {
                    val isUnmasked = !isMaskPixel(maskPx[idx])
                    if (!isUnmasked) continue
                    var touchesMask = false
                    checkT@ for (dy in -r..r) for (dx in -r..r) {
                        val ni = idx + dy * w + dx
                        if (isMaskPixel(maskPx[ni])) {
                            val dd = abs(dx) + abs(dy)
                            if (dd <= seamWidth) { touchesMask = true; break@checkT }
                        }
                    }
                    if (!touchesMask) continue
                    falloff = 0.45f
                }
                var sr = 0f; var sg = 0f; var sb = 0f; var wt = 0f
                for (dy in -r..r) {
                    val offRow = idx + dy * w
                    for (dx in -r..r) {
                        val dist11 = sqrt((dx * dx + dy * dy).toFloat())
                        val wd = 1f / (1f + dist11)
                        val c = compPx[offRow + dx]
                        sr += Color.red(c) * wd
                        sg += Color.green(c) * wd
                        sb += Color.blue(c) * wd
                        wt += wd
                    }
                }
                val base = compPx[idx]
                val mixA = 1f - falloff * 0.65f
                val mixB = falloff * 0.65f
                val fr = Color.red(base) * mixA + (sr / wt) * mixB
                val fg = Color.green(base) * mixA + (sg / wt) * mixB
                val fb = Color.blue(base) * mixA + (sb / wt) * mixB
                outPx[idx] = Color.rgb(
                    fr.coerceIn(0f, 255f).toInt(),
                    fg.coerceIn(0f, 255f).toInt(),
                    fb.coerceIn(0f, 255f).toInt()
                )
            }
        }
        val res = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        res.setPixels(outPx, 0, w, 0, 0, w, h)
        return res
    }
}
