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
import android.util.Log

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class ObjectRemover(context: Context) : TFLiteModel(context, "lama_dilated-tflite-float.tflite") {

    private val modelWidth = 512
    private val modelHeight = 512
    private val nativeProcessor = NativeProcessor()

    // Reusable inference buffers (the model graph is a fixed 512x512).
    private val imgBuf = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * 3).order(ByteOrder.nativeOrder())
    private val maskBuf = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * 1).order(ByteOrder.nativeOrder())
    private val outputBuf = ByteBuffer.allocateDirect(4 * modelWidth * modelHeight * 3).order(ByteOrder.nativeOrder())

    private val imageInputIndex: Int
    private val maskInputIndex: Int
    private val imageIsNchw: Boolean
    private val maskIsNchw: Boolean

    init {
        val inputCount = interpreter.inputTensorCount
        require(inputCount == 2) { "LaMa model must have image and mask inputs" }
        imageInputIndex = (0 until inputCount).firstOrNull { index ->
            val name = interpreter.getInputTensor(index).name().lowercase()
            name.contains("painted") || name.contains("image")
        } ?: 0
        maskInputIndex = (0 until inputCount).firstOrNull { index ->
            interpreter.getInputTensor(index).name().lowercase().contains("mask")
        } ?: (1 - imageInputIndex)
        require(imageInputIndex != maskInputIndex) { "LaMa image and mask inputs are ambiguous" }

        val imageShape = interpreter.getInputTensor(imageInputIndex).shape()
        val maskShape = interpreter.getInputTensor(maskInputIndex).shape()
        imageIsNchw = imageShape.size == 4 && imageShape[1] == 3
        maskIsNchw = maskShape.size == 4 && maskShape[1] == 1
    }

    suspend fun removeObject(image: Bitmap, mask: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        val w = image.width
        val h = image.height

        val safeImage = image.copy(image.config ?: Bitmap.Config.ARGB_8888, true)
        val safeMask = mask.copy(Bitmap.Config.ARGB_8888, true)

        // Scale-aware dilation so fur wisps, anti-aliased brush edges and the soft
        // shadow halo around the object are fully inside the hole. The old fixed
        // 2-4px dilation left a visible ghost outline of the removed object.
        val dilatedMask = dilateMask(safeMask, dilationRadiusPx(max(w, h)))
        safeMask.recycle()

        val maskPx = IntArray(w * h)
        dilatedMask.getPixels(maskPx, 0, w, 0, 0, w, h)
        val origPx = IntArray(w * h)
        safeImage.getPixels(origPx, 0, w, 0, 0, w, h)

        val bounds = maskBounds(maskPx, w, h)
        if (bounds == null) {
            dilatedMask.recycle()
            return@withContext safeImage
        }

        val aiInpainted: Bitmap = if (max(w, h) <= modelWidth) {
            // Small photos already run at (near) native resolution in one pass.
            runLaMaOnBitmap(safeImage, dilatedMask)
        } else {
            // KEY FIX: run the model on native-resolution 512x512 tiles that
            // overlap around the hole, instead of squashing the whole image to
            // 512x512 and stretching the answer back. Texture is generated 1:1,
            // so the result no longer looks like an upscaled blur.
            inpaintWithTiles(origPx, maskPx, w, h, bounds)
        }

        val rawInpainted: Bitmap = if (isDegenerateOutput(aiInpainted, origPx, maskPx)) {
            Log.w("LaMa", "AI output looks degenerate; using native fallback")
            aiInpainted.recycle()
            val fallback = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val status = nativeProcessor.processImage(safeImage, dilatedMask, fallback)
            if (status != 0) {
                fallback.recycle()
                safeImage.recycle()
                dilatedMask.recycle()
                throw IllegalStateException("Native inpainting failed with status $status")
            }
            fallback
        } else {
            aiInpainted
        }

        val finalResult = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val status = nativeProcessor.seamlessComposite(safeImage, rawInpainted, dilatedMask, finalResult)
        val resultBitmap = if (status == 0) {
            finalResult
        } else {
            Log.w("ObjectRemover", "Native seamlessComposite failed ($status); using alpha blend fallback")
            finalResult.recycle()
            val alphaMask = buildCompositeAlphaMask(dilatedMask, 12f)
            val fallback = legacySeamlessComposite(safeImage, rawInpainted, alphaMask)
            alphaMask.recycle()
            fallback
        }

        rawInpainted.recycle()
        safeImage.recycle()
        dilatedMask.recycle()

        resultBitmap
    }

    // ---------------------------------------------------------------------
    // Mask utilities
    // ---------------------------------------------------------------------

    // Decide "selected" by LUMINANCE only, never by alpha. The mask is a binary image:
    // opaque white = selected (to inpaint), opaque black = keep. Checking alpha would
    // treat the opaque-black background (alpha 255) as selected and flood the whole
    // image into the model as a hole, producing a white result.
    private fun isMaskPixel(c: Int): Boolean {
        return Color.red(c) > 127 || Color.green(c) > 127 || Color.blue(c) > 127
    }

    private fun dilationRadiusPx(maxDim: Int): Int = (maxDim / 100).coerceIn(4, 28)

    /**
     * Separable box dilation, O(w*h) regardless of radius. Replaces the old
     * fixed-round 3x3 dilation whose radius capped out at ~4px on big photos.
     */
    private fun dilateMask(mask: Bitmap, radius: Int): Bitmap {
        val w = mask.width
        val h = mask.height
        val n = w * h
        val px = IntArray(n)
        mask.getPixels(px, 0, w, 0, 0, w, h)
        val src = BooleanArray(n) { isMaskPixel(px[it]) }
        val tmp = BooleanArray(n)
        val out = BooleanArray(n)

        // Horizontal pass with a sliding window count
        for (y in 0 until h) {
            val row = y * w
            var count = 0
            for (k in 0..min(radius, w - 1)) if (src[row + k]) count++
            for (x in 0 until w) {
                tmp[row + x] = count > 0
                val add = x + radius + 1
                if (add < w && src[row + add]) count--
                val rem = x - radius
                if (rem >= 0 && src[row + rem]) count--
            }
        }
        // Vertical pass with a sliding window count
        for (x in 0 until w) {
            var count = 0
            for (k in 0..min(radius, h - 1)) if (tmp[k * w + x]) count++
            for (y in 0 until h) {
                out[y * w + x] = count > 0
                val add = y + radius + 1
                if (add < h && tmp[add * w + x]) count--
                val rem = y - radius
                if (rem >= 0 && tmp[rem * w + x]) count--
            }
        }

        val outPx = IntArray(n)
        for (i in 0 until n) outPx[i] = if (out[i]) Color.WHITE else Color.BLACK
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(outPx, 0, w, 0, 0, w, h)
        return result
    }

    /** Bounding box of the mask, exclusive right/bottom; null when empty. */
    private fun maskBounds(maskPx: IntArray, w: Int, h: Int): Rect? {
        var minX = w
        var minY = h
        var maxX = -1
        var maxY = -1
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                if (isMaskPixel(maskPx[row + x])) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        if (maxX < 0) return null
        return Rect(minX, minY, maxX + 1, maxY + 1)
    }

    // ---------------------------------------------------------------------
    // Tiled native-resolution inference
    // ---------------------------------------------------------------------

    private data class TileWindow(val x: Int, val y: Int)

    /**
     * Inpaints the masked area by running the fixed 512x512 LaMa graph over a
     * grid of native-resolution 512px windows that overlap around the hole.
     * Each tile sees real surrounding context at 1:1 scale; outputs are merged
     * with an edge-fade weight so tile seams are invisible.
     */
    private fun inpaintWithTiles(
        origPx: IntArray, maskPx: IntArray, w: Int, h: Int, bounds: Rect
    ): Bitmap {
        val t = modelWidth
        val started = System.currentTimeMillis()

        val accR = FloatArray(w * h)
        val accG = FloatArray(w * h)
        val accB = FloatArray(w * h)
        val accW = FloatArray(w * h)

        // If the whole hole (plus context margin) fits into one window, a single
        // pass is enough. Otherwise cover it with an overlapping tile grid.
        val margin = 48
        val windows: List<TileWindow> = if (
            bounds.width() + 2 * margin <= t && bounds.height() + 2 * margin <= t
        ) {
            val cx = bounds.centerX()
            val cy = bounds.centerY()
            val tx = if (w >= t) (cx - t / 2).coerceIn(0, w - t) else (w - t) / 2
            val ty = if (h >= t) (cy - t / 2).coerceIn(0, h - t) else (h - t) / 2
            listOf(TileWindow(tx, ty))
        } else {
            val stride = t - 96 // 96px overlap = 2x the 48px edge fade
            val xs = planTileStarts(w, t, stride)
            val ys = planTileStarts(h, t, stride)
            ys.flatMap { ty -> xs.map { tx -> TileWindow(tx, ty) } }
        }

        var ranTiles = 0
        for (win in windows) {
            if (!windowTouchesMask(maskPx, w, win.x, win.y, t)) continue
            val tileStart = System.currentTimeMillis()
            ranTiles++

            val tileImgPx = windowPixels(origPx, w, h, win.x, win.y, t, t)
            val tileMaskPx = windowPixels(maskPx, w, h, win.x, win.y, t, t)
            val tileImg = bitmapFromPixels(tileImgPx, t, t)
            val tileMask = bitmapFromPixels(tileMaskPx, t, t)
            val tileOut = runLaMaCore(tileImg, tileMask)
            tileImg.recycle()
            tileMask.recycle()

            val outPx = IntArray(t * t)
            tileOut.getPixels(outPx, 0, t, 0, 0, t, t)
            tileOut.recycle()

            val fade = 48f
            for (j in 0 until t) {
                val iy = win.y + j
                if (iy < 0 || iy >= h) continue
                val wy = axisWeight(j, t, fade, fadeLeft = win.y > 0, fadeRight = win.y + t < h)
                val row = iy * w
                val outRow = j * t
                for (i in 0 until t) {
                    val ix = win.x + i
                    if (ix < 0 || ix >= w) continue
                    val idx = row + ix
                    if (!isMaskPixel(maskPx[idx])) continue
                    val wx = axisWeight(i, t, fade, fadeLeft = win.x > 0, fadeRight = win.x + t < w)
                    val wgt = wx * wy
                    if (wgt <= 0f) continue
                    val c = outPx[outRow + i]
                    accR[idx] += Color.red(c) * wgt
                    accG[idx] += Color.green(c) * wgt
                    accB[idx] += Color.blue(c) * wgt
                    accW[idx] += wgt
                }
            }
            Log.d("LaMa", "tile ${win.x},${win.y} took ${System.currentTimeMillis() - tileStart} ms")
        }

        // Composite: original everywhere, blended model output inside the mask.
        val outFull = IntArray(w * h)
        System.arraycopy(origPx, 0, outFull, 0, w * h)
        for (i in outFull.indices) {
            val aw = accW[i]
            if (aw > 0f) {
                val r = (accR[i] / aw).toInt().coerceIn(0, 255)
                val g = (accG[i] / aw).toInt().coerceIn(0, 255)
                val b = (accB[i] / aw).toInt().coerceIn(0, 255)
                outFull[i] = Color.rgb(r, g, b)
            }
        }
        Log.d("LaMa", "tiled inpaint: $ranTiles/${windows.size} windows in ${System.currentTimeMillis() - started} ms")

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(outFull, 0, w, 0, 0, w, h)
        return result
    }

    private fun planTileStarts(length: Int, tileSize: Int, stride: Int): List<Int> {
        if (length <= tileSize) return listOf(0) // single reflect-padded window
        val starts = mutableListOf<Int>()
        var s = 0
        while (s + tileSize < length) {
            starts.add(s)
            s += stride
        }
        val last = length - tileSize
        if (starts.isEmpty() || starts.last() < last) starts.add(last)
        return starts
    }

    /**
     * Weight of a pixel along one axis. Edges that touch the image border keep
     * full weight (they are the only coverage there); interior edges fade over
     * [fade] px so overlapping tiles blend smoothly.
     */
    private fun axisWeight(pos: Int, size: Int, fade: Float, fadeLeft: Boolean, fadeRight: Boolean): Float {
        var d = Int.MAX_VALUE
        if (fadeLeft) d = min(d, pos)
        if (fadeRight) d = min(d, size - 1 - pos)
        if (d == Int.MAX_VALUE) return 1f
        if (d >= fade) return 1f
        val t = d / fade
        return t * t * (3f - 2f * t) // smoothstep
    }

    private fun windowTouchesMask(maskPx: IntArray, w: Int, x0: Int, y0: Int, t: Int): Boolean {
        val h = maskPx.size / w
        val xEnd = min(x0 + t, w)
        val yEnd = min(y0 + t, h)
        for (y in max(0, y0) until yEnd) {
            val row = y * w
            for (x in max(0, x0) until xEnd) {
                if (isMaskPixel(maskPx[row + x])) return true
            }
        }
        return false
    }

    /** Crops a window (reflect-padded when it extends past the image). */
    private fun windowPixels(src: IntArray, w: Int, h: Int, x0: Int, y0: Int, tw: Int, th: Int): IntArray {
        val out = IntArray(tw * th)
        for (j in 0 until th) {
            var sy = y0 + j
            if (sy < 0) sy = -sy
            if (sy >= h) sy = 2 * (h - 1) - sy
            val srcRow = sy * w
            val dstRow = j * tw
            for (i in 0 until tw) {
                var sx = x0 + i
                if (sx < 0) sx = -sx
                if (sx >= w) sx = 2 * (w - 1) - sx
                out[dstRow + i] = src[srcRow + sx]
            }
        }
        return out
    }

    private fun bitmapFromPixels(px: IntArray, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    // ---------------------------------------------------------------------
    // Model runner
    // ---------------------------------------------------------------------

    /** Runs the 512x512 graph once and returns the 512x512 result. */
    private fun runLaMaCore(input: Bitmap, inputMask: Bitmap): Bitmap {
        require(input.width == modelWidth && input.height == modelHeight) {
            "runLaMaCore expects a ${modelWidth}x${modelHeight} tile"
        }
        fillImageBuffer(input, inputMask, imgBuf, imageIsNchw)
        fillMaskBuffer(inputMask, maskBuf, maskIsNchw)
        imgBuf.rewind()
        maskBuf.rewind()
        outputBuf.rewind()

        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(imgBuf, maskBuf).also { inputs ->
                inputs[imageInputIndex] = imgBuf
                inputs[maskInputIndex] = maskBuf
            },
            mapOf(0 to outputBuf)
        )

        val outputShape = interpreter.getOutputTensor(0).shape()
        val isNchwOutput = outputShape.size == 4 && outputShape[1] == 3
        return convertOutputToBitmap(outputBuf, modelWidth, modelHeight, isNchwOutput)
    }

    /** Whole-image path for photos that already fit the 512x512 graph. */
    private fun runLaMaOnBitmap(input: Bitmap, inputMask: Bitmap): Bitmap {
        val w = input.width
        val h = input.height

        val (squareImg, squareMask, padL, padT, padR, padB, squareSize) =
            padToSquareReflect(input, inputMask)

        val scaledImg = if (squareSize == modelWidth) squareImg
        else Bitmap.createScaledBitmap(squareImg, modelWidth, modelHeight, true)
        val scaledMask = if (squareSize == modelWidth) squareMask
        else Bitmap.createScaledBitmap(squareMask, modelWidth, modelHeight, true)
        if (scaledImg !== squareImg) squareImg.recycle()
        if (scaledMask !== squareMask) squareMask.recycle()

        val rawSquare = runLaMaCore(scaledImg, scaledMask)
        scaledImg.recycle()
        scaledMask.recycle()

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

    /**
     * Detects a broken model output (the old green/magenta checkerboard or a
     * collapsed flat fill) *inside the masked region only*. The previous
     * isMostlyWhite check sampled the whole image and false-triggered on bright
     * scenes (white walls, mattresses), throwing away good results.
     */
    private fun isDegenerateOutput(ai: Bitmap, origPx: IntArray, maskPx: IntArray): Boolean {
        val w = ai.width
        val h = ai.height
        val aiPx = IntArray(w * h)
        ai.getPixels(aiPx, 0, w, 0, 0, w, h)

        var n = 0
        var weird = 0
        var aiSum = 0.0
        var aiSqSum = 0.0
        var origSum = 0.0
        var origSqSum = 0.0
        for (i in aiPx.indices) {
            if (!isMaskPixel(maskPx[i])) continue
            n++
            val c = aiPx[i]
            val r = Color.red(c)
            val g = Color.green(c)
            val b = Color.blue(c)
            if ((g > r + 40 && g > b + 40) || (r > g + 60 && b > g + 60)) weird++
            val lum = (r + g + b) / 3.0
            aiSum += lum
            aiSqSum += lum * lum
            val oc = origPx[i]
            val olum = (Color.red(oc) + Color.green(oc) + Color.blue(oc)) / 3.0
            origSum += olum
            origSqSum += olum * olum
        }
        if (n == 0) return false
        if (weird.toFloat() / n > 0.25f) return true

        val aiMean = aiSum / n
        val aiVar = max(0.0, aiSqSum / n - aiMean * aiMean)
        val origMean = origSum / n
        val origVar = max(0.0, origSqSum / n - origMean * origMean)
        // Collapsed to a flat color while the original region had real texture.
        return aiVar < 4.0 && origVar > 150.0
    }

    private fun fillImageBuffer(img: Bitmap, mask: Bitmap, imgBuf: ByteBuffer, isNchw: Boolean) {
        val w = img.width
        val h = img.height
        val total = w * h
        val iPx = IntArray(total)
        val mPx = IntArray(total)
        img.getPixels(iPx, 0, w, 0, 0, w, h)
        mask.getPixels(mPx, 0, w, 0, 0, w, h)
        imgBuf.rewind()

        if (isNchw) {
            for (i in 0 until total) {
                val isMasked = isMaskPixel(mPx[i])
                imgBuf.putFloat(if (isMasked) 0f else Color.red(iPx[i]) / 255f)
            }
            for (i in 0 until total) {
                val isMasked = isMaskPixel(mPx[i])
                imgBuf.putFloat(if (isMasked) 0f else Color.green(iPx[i]) / 255f)
            }
            for (i in 0 until total) {
                val isMasked = isMaskPixel(mPx[i])
                imgBuf.putFloat(if (isMasked) 0f else Color.blue(iPx[i]) / 255f)
            }
        } else {
            for (i in 0 until total) {
                val isMasked = isMaskPixel(mPx[i])
                val c = iPx[i]
                imgBuf.putFloat(if (isMasked) 0f else Color.red(c) / 255f)
                imgBuf.putFloat(if (isMasked) 0f else Color.green(c) / 255f)
                imgBuf.putFloat(if (isMasked) 0f else Color.blue(c) / 255f)
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

        val floatBuf = buf.asFloatBuffer()
        val floats = FloatArray(floatBuf.remaining())
        floatBuf.get(floats)

        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        val checkStep = maxOf(1, floats.size / 1000)
        for (i in floats.indices step checkStep) {
            val v = floats[i]
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }

        val mode = when {
            maxVal > 2.0f -> 0 // In [0, 255]
            minVal < -0.1f -> 1 // In [-1, 1]
            else -> 2 // In [0, 1]
        }

        fun toByte(v: Float): Int {
            return when (mode) {
                0 -> v.coerceIn(0f, 255f).toInt()
                1 -> ((v + 1f) * 127.5f).coerceIn(0f, 255f).toInt()
                else -> (v * 255f).coerceIn(0f, 255f).toInt()
            }
        }

        if (isNchw) {
            val rOffset = 0
            val gOffset = total
            val bOffset = total * 2
            for (i in 0 until total) {
                val r = toByte(floats[rOffset + i])
                val g = toByte(floats[gOffset + i])
                val b = toByte(floats[bOffset + i])
                out[i] = Color.rgb(r, g, b)
            }
        } else {
            for (i in 0 until total) {
                val r = toByte(floats[i * 3])
                val g = toByte(floats[i * 3 + 1])
                val b = toByte(floats[i * 3 + 2])
                out[i] = Color.rgb(r, g, b)
            }
        }
        val bm = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bm.setPixels(out, 0, w, 0, 0, w, h)
        return bm
    }

    // ---------------------------------------------------------------------
    // Reflection padding to square (small-image path)
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
        val sqImg = bitmapFromPixels(sqImgPx, side, side)
        val sqMask = bitmapFromPixels(sqMaskPx, side, side)
        return Tuple7(sqImg, sqMask, padL, padT, padR, padB, side)
    }

    private data class Tuple7<A, B, C, D, E, F, G>(
        val a: A, val b: B, val c: C, val d: D, val e: E, val f: F, val g: G
    )

    // ---------------------------------------------------------------------
    // Composite fallbacks (only used if the native path fails)
    // ---------------------------------------------------------------------

    private fun buildCompositeAlphaMask(mask: Bitmap, blurRadius: Float): Bitmap {
        val w = mask.width
        val h = mask.height
        val total = w * h

        val srcPixels = IntArray(total)
        mask.getPixels(srcPixels, 0, w, 0, 0, w, h)

        val alphaPixels = IntArray(total)
        val isMasked = BooleanArray(total)
        for (i in 0 until total) {
            val masked = isMaskPixel(srcPixels[i])
            isMasked[i] = masked
            alphaPixels[i] = if (masked) Color.WHITE else Color.TRANSPARENT
        }

        val binaryAlphaBmp = bitmapFromPixels(alphaPixels, w, h)

        val radius = blurRadius.coerceAtLeast(1f)
        val blurredBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(blurredBmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            maskFilter = BlurMaskFilter(radius, BlurMaskFilter.Blur.NORMAL)
        }
        canvas.drawBitmap(binaryAlphaBmp, 0f, 0f, paint)
        binaryAlphaBmp.recycle()

        val blurredPixels = IntArray(total)
        blurredBmp.getPixels(blurredPixels, 0, w, 0, 0, w, h)
        blurredBmp.recycle()

        for (i in 0 until total) {
            if (isMasked[i]) {
                // Interior of selection stays 100% opaque (alpha = 255)
                blurredPixels[i] = Color.WHITE
            } else {
                // Feathered outward edge: keep alpha from blur
                val a = Color.alpha(blurredPixels[i])
                blurredPixels[i] = Color.argb(a, 255, 255, 255)
            }
        }

        return bitmapFromPixels(blurredPixels, w, h)
    }

    private fun legacySeamlessComposite(
        original: Bitmap, inpainted: Bitmap, alphaMask: Bitmap
    ): Bitmap {
        val w = original.width; val h = original.height
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(out)
        c.drawBitmap(original, 0f, 0f, null)

        val maskedInpaint = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val ci = Canvas(maskedInpaint)
        ci.drawBitmap(inpainted, 0f, 0f, null)
        val dstIn = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        }
        ci.drawBitmap(alphaMask, 0f, 0f, dstIn)
        c.drawBitmap(maskedInpaint, 0f, 0f, null)
        maskedInpaint.recycle()
        return out
    }
}
