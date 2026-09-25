package com.example.magicimagepro.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
import java.io.File
import java.nio.FloatBuffer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Object removal powered by the original big-lama inpainting model
 * (SAIC LaMa, Places2 checkpoint), executed with ONNX Runtime.
 *
 * The ONNX export (Carve/LaMa-ONNX) has a fixed 512x512 input:
 *   image: float32 [N,512,512,3] in [0,1]  (NHWC)
 *   mask : float32 [N,512,512,1] in {0,1}  (NHWC, 1 = hole)
 *   out  : float32 [N,3,512,512] in [0,255] (NCHW)
 *
 * High-resolution photos are handled by running the fixed-size graph over
 * native-resolution 512px tiles that overlap around the hole, then merging
 * the results with an edge-fade blend. Texture is generated 1:1, so the
 * result stays as sharp as the photo it came from.
 */
class ObjectRemover(context: Context) {

    private val modelWidth = 512
    private val modelHeight = 512
    private val nativeProcessor = NativeProcessor()

    /**
     * Reason the last [removeObject] run used the basic (OpenCV) repair instead
     * of the AI fill, or null when the AI fill was used. Surfaced to the user
     * so silent degradation is impossible.
     */
    var lastRunDegraded: String? = null
        private set

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val imageInputName: String
    private val maskInputName: String

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        val modelFile = ensureModelFile(context)
        session = ortEnv.createSession(modelFile.absolutePath, options)

        val names = session.inputNames.toList()
        require(names.size == 2) { "LaMa model must have image and mask inputs" }
        imageInputName = names.firstOrNull { it.contains("image") || it.contains("painted") } ?: names[0]
        maskInputName = names.firstOrNull { it.contains("mask") } ?: names[1]
        require(imageInputName != maskInputName) { "LaMa image and mask inputs are ambiguous" }
        Log.d("LaMa", "ONNX session ready; inputs: image='$imageInputName' mask='$maskInputName'")
    }

    fun close() {
        session.close()
    }

    private fun ensureModelFile(context: Context): File {
        val f = File(context.filesDir, "big_lama.onnx")
        if (f.exists() && f.length() > 0L) return f
        context.assets.open("big_lama.onnx").use { input ->
            val tmp = File.createTempFile("big_lama", ".onnx", context.cacheDir)
            tmp.outputStream().use { input.copyTo(it) }
            if (!tmp.renameTo(f)) {
                tmp.copyTo(f, overwrite = true)
                tmp.delete()
            }
        }
        return f
    }

    suspend fun removeObject(image: Bitmap, mask: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        lastRunDegraded = null
        val w = image.width
        val h = image.height

        val safeImage = image.copy(image.config ?: Bitmap.Config.ARGB_8888, true)
        val safeMask = mask.copy(Bitmap.Config.ARGB_8888, true)

        // Scale-aware dilation so fur wisps, anti-aliased brush edges and the soft
        // shadow halo around the object are fully inside the hole.
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
            // Run the model on native-resolution 512x512 tiles that overlap
            // around the hole, instead of squashing the whole image to 512x512.
            inpaintWithTiles(origPx, maskPx, w, h, bounds)
        }

        // Never return a no-op: if the model output is degenerate (checkerboard /
        // flat fill) or barely changed the masked pixels, fall back to the native
        // OpenCV inpainter so "Remove object" always does something.
        val degenerate = isDegenerateOutput(aiInpainted, origPx, maskPx)
        val coverage = maskedCoverage(aiInpainted, origPx, maskPx)
        val rawInpainted: Bitmap = if (degenerate || coverage < 0.55f) {
            lastRunDegraded = if (degenerate) "corrupted output" else "empty fill (model echoed input)"
            Log.w("LaMa", "AI output unusable ($lastRunDegraded); using native fallback")
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

        // Anti-tell finishing pass: phone photos carry sensor grain, and the AI
        // fill comes out smoother than its surroundings. Re-inject noise matched
        // to the photo's own grain level. (The native basic-repair path already
        // applies its own grain in C++, so only grain the AI path here.)
        if (lastRunDegraded == null) {
            val grainStd = estimateSensorGrain(origPx, maskPx, w, h, bounds)
            if (grainStd > 0.3f) {
                applyGrain(resultBitmap, maskPx, w, h, grainStd)
            }
        }

        rawInpainted.recycle()
        safeImage.recycle()
        dilatedMask.recycle()

        resultBitmap
    }

    // ---------------------------------------------------------------------
    // Grain matching
    // ---------------------------------------------------------------------

    /**
     * Estimates the photo's sensor-grain strength near the hole: the standard
     * deviation of high-frequency luminance on unmasked pixels around the
     * masked bounds.
     */
    private fun estimateSensorGrain(
        origPx: IntArray, maskPx: IntArray, w: Int, h: Int, bounds: Rect
    ): Float {
        val pad = 48
        val l = max(0, bounds.left - pad)
        val t = max(0, bounds.top - pad)
        val r = min(w, bounds.right + pad)
        val b = min(h, bounds.bottom + pad)
        if (r - l < 4 || b - t < 4) return 0f

        var sum = 0.0
        var sqSum = 0.0
        var n = 0
        for (y in t + 1 until b - 1) {
            val row = y * w
            for (x in l + 1 until r - 1) {
                val idx = row + x
                if (isMaskPixel(maskPx[idx])) continue
                // 3x3 box blur luminance as the local mean
                var lumSum = 0.0
                for (dy in -1..1) {
                    val rr = (y + dy) * w
                    for (dx in -1..1) {
                        val c = origPx[rr + x + dx]
                        lumSum += (Color.red(c) + Color.green(c) + Color.blue(c)) / 3.0
                    }
                }
                val c = origPx[idx]
                val lum = (Color.red(c) + Color.green(c) + Color.blue(c)) / 3.0
                val highFreq = lum - lumSum / 9.0
                sum += highFreq
                sqSum += highFreq * highFreq
                n++
            }
        }
        if (n < 64) return 0f
        val mean = sum / n
        return sqrt(max(0.0, sqSum / n - mean * mean)).toFloat().coerceAtMost(6f)
    }

    /**
     * Adds luminance noise with the given standard deviation to masked pixels
     * (same offset on R/G/B, like real sensor noise).
     */
    private fun applyGrain(bmp: Bitmap, maskPx: IntArray, w: Int, h: Int, std: Float) {
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        val amp = std * 1.732f * 0.85f // uniform[-a,a] has std a/sqrt(3)
        val rng = java.util.Random(System.nanoTime())
        for (i in px.indices) {
            if (!isMaskPixel(maskPx[i])) continue
            val noise = ((rng.nextFloat() * 2f - 1f) * amp)
            val c = px[i]
            px[i] = Color.rgb(
                (Color.red(c) + noise).toInt().coerceIn(0, 255),
                (Color.green(c) + noise).toInt().coerceIn(0, 255),
                (Color.blue(c) + noise).toInt().coerceIn(0, 255)
            )
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
    }

    // ---------------------------------------------------------------------
    // Mask utilities
    // ---------------------------------------------------------------------

    // White = selected (to inpaint), black = keep. Luminance only.
    private fun isMaskPixel(c: Int): Boolean {
        return Color.red(c) > 127 || Color.green(c) > 127 || Color.blue(c) > 127
    }

    private fun dilationRadiusPx(maxDim: Int): Int = (maxDim / 100).coerceIn(4, 28)

    /**
     * Separable box dilation, O(w*h) regardless of radius.
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

        // Horizontal pass with a sliding window count. Window at x covers
        // [x-radius, x+radius]: entering pixel (x+radius+1) must INCREMENT the
        // count, leaving pixel (x-radius) decrements it.
        for (y in 0 until h) {
            val row = y * w
            var count = 0
            for (k in 0..min(radius, w - 1)) if (src[row + k]) count++
            for (x in 0 until w) {
                tmp[row + x] = count > 0
                val add = x + radius + 1
                if (add < w && src[row + add]) count++
                val rem = x - radius
                if (rem >= 0 && src[row + rem]) count--
            }
        }
        // Vertical pass with a sliding window count (same invariants)
        for (x in 0 until w) {
            var count = 0
            for (k in 0..min(radius, h - 1)) if (tmp[k * w + x]) count++
            for (y in 0 until h) {
                out[y * w + x] = count > 0
                val add = y + radius + 1
                if (add < h && tmp[add * w + x]) count++
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
    // ONNX model runner
    // ---------------------------------------------------------------------

    /** Runs the 512x512 graph once and returns the 512x512 result. */
    private fun runLaMaCore(input: Bitmap, inputMask: Bitmap): Bitmap {
        val t = modelWidth
        require(input.width == t && input.height == t) {
            "runLaMaCore expects a ${t}x${t} tile"
        }
        val total = t * t

        val px = IntArray(total)
        input.getPixels(px, 0, t, 0, 0, t, t)
        val mpx = IntArray(total)
        inputMask.getPixels(mpx, 0, t, 0, 0, t, t)

        val imgArr = FloatArray(total * 3)
        val maskArr = FloatArray(total)
        val rOffset = 0
        val gOffset = total
        val bOffset = total * 2
        for (i in 0 until total) {
            val c = px[i]
            imgArr[rOffset + i] = Color.red(c) / 255f
            imgArr[gOffset + i] = Color.green(c) / 255f
            imgArr[bOffset + i] = Color.blue(c) / 255f
            maskArr[i] = if (isMaskPixel(mpx[i])) 1f else 0f
        }

        val imgTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(imgArr), longArrayOf(1, 3, t.toLong(), t.toLong())
        )
        val maskTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(maskArr), longArrayOf(1, 1, t.toLong(), t.toLong())
        )
        try {
            session.run(mapOf(imageInputName to imgTensor, maskInputName to maskTensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                return convertOutputToBitmap(outTensor, t)
            }
        } finally {
            imgTensor.close()
            maskTensor.close()
        }
    }

    private fun convertOutputToBitmap(outTensor: OnnxTensor, size: Int): Bitmap {
        val total = size * size
        val buf = outTensor.floatBuffer
        val floats = FloatArray(buf.remaining())
        buf.get(floats)

        val shape = outTensor.info.shape
        val isNchw = shape.size == 4 && shape[1] == 3L

        // Detect output scale: this export emits [0,255]; be tolerant of
        // exports that emit [0,1] or [-1,1].
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        val checkStep = maxOf(1, floats.size / 1000)
        for (i in floats.indices step checkStep) {
            val v = floats[i]
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }
        val scale = when {
            maxVal > 2.0f -> 1f        // already [0,255]
            minVal < -0.1f -> 127.5f   // [-1,1]
            else -> 255f               // [0,1]
        }

        val out = IntArray(total)
        if (isNchw) {
            val rOffset = 0
            val gOffset = total
            val bOffset = total * 2
            for (i in 0 until total) {
                val r = (floats[rOffset + i] * scale).toInt().coerceIn(0, 255)
                val g = (floats[gOffset + i] * scale).toInt().coerceIn(0, 255)
                val b = (floats[bOffset + i] * scale).toInt().coerceIn(0, 255)
                out[i] = Color.rgb(r, g, b)
            }
        } else {
            for (i in 0 until total) {
                val r = (floats[i * 3] * scale).toInt().coerceIn(0, 255)
                val g = (floats[i * 3 + 1] * scale).toInt().coerceIn(0, 255)
                val b = (floats[i * 3 + 2] * scale).toInt().coerceIn(0, 255)
                out[i] = Color.rgb(r, g, b)
            }
        }
        val bm = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        bm.setPixels(out, 0, size, 0, 0, size, size)
        return bm
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

    // ---------------------------------------------------------------------
    // Output sanity checks
    // ---------------------------------------------------------------------

    /**
     * Detects a broken model output (green/magenta checkerboard or collapsed
     * flat fill) inside the masked region only. The old whole-image "mostly
     * white" check false-triggered on bright scenes and threw away good results.
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
        return aiVar < 4.0 && origVar > 150.0
    }

    /**
     * Fraction of masked pixels the model actually changed. A healthy
     * generative fill rewrites essentially the whole hole.
     */
    private fun maskedCoverage(ai: Bitmap, origPx: IntArray, maskPx: IntArray): Float {
        val w = ai.width
        val h = ai.height
        val aiPx = IntArray(w * h)
        ai.getPixels(aiPx, 0, w, 0, 0, w, h)
        var n = 0
        var changed = 0
        for (i in aiPx.indices) {
            if (!isMaskPixel(maskPx[i])) continue
            n++
            val a = aiPx[i]
            val o = origPx[i]
            val d = maxOf(
                abs(Color.red(a) - Color.red(o)),
                abs(Color.green(a) - Color.green(o)),
                abs(Color.blue(a) - Color.blue(o))
            )
            if (d >= 10) changed++
        }
        return if (n == 0) 1f else changed.toFloat() / n
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
                blurredPixels[i] = Color.WHITE
            } else {
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
