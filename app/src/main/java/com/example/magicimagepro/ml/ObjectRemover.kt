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
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import java.io.File
import java.nio.FloatBuffer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
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
 * High-resolution photos are handled by cropping one square window around the
 * hole (the object plus a ring of context) and squeezing that window into the
 * graph's single 512x512 field of view, then mapping the fill back to native
 * resolution. Showing the model the whole hole at once keeps the mask a small
 * fraction of its input, so it reads the real surroundings and extends them.
 * Feeding it crops that each cover only part of the hole made it invent
 * content instead, leaving a dark smudged blob where a large object had been.
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
        if ((f.exists()) && (f.length() > 0L)) return f
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

        val aiInpainted: Bitmap = inpaintWithContext(origPx, maskPx, w, h, bounds)

        // Never return a no-op: if the model output is degenerate (checkerboard /
        // flat fill) or barely changed the masked pixels, fall back to the native
        // OpenCV inpainter so "Remove object" always does something.
        val degenerate = isDegenerateOutput(aiInpainted, origPx, maskPx)
        val coverage = maskedCoverage(aiInpainted, origPx, maskPx)
        val rawInpainted: Bitmap = if (degenerate || coverage < 0.55f) {
            lastRunDegraded = if (degenerate) "corrupted output" else "empty fill (model echoed input)"
            Log.w("LaMa", "AI output unusable ($lastRunDegraded); using native fallback")
            aiInpainted.recycle()
            val fallback = createBitmap(w, h, Bitmap.Config.ARGB_8888)
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

        // The graph runs at 512x512, so a big hole comes back structurally
        // right but smoothly rendered. Borrow the photo's own fine detail for
        // it, otherwise the fill is the one soft patch in a grainy photo. Only
        // meaningful for the AI fill (the basic repair grains itself in C++).
        val polished: Bitmap = if (lastRunDegraded == null) {
            val refined = createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val refineStatus = nativeProcessor.refineFill(safeImage, rawInpainted, dilatedMask, refined)
            if (refineStatus == 0) {
                rawInpainted.recycle()
                refined
            } else {
                Log.w("ObjectRemover", "Native refineFill failed ($refineStatus); using raw model output")
                refined.recycle()
                rawInpainted
            }
        } else {
            rawInpainted
        }

        val finalResult = createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val status = nativeProcessor.seamlessComposite(safeImage, polished, dilatedMask, finalResult)
        val resultBitmap = if (status == 0) {
            finalResult
        } else {
            Log.w("ObjectRemover", "Native seamlessComposite failed ($status); using alpha blend fallback")
            finalResult.recycle()
            val alphaMask = buildCompositeAlphaMask(dilatedMask)
            val fallback = legacySeamlessComposite(safeImage, polished, alphaMask)
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

        polished.recycle()
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
        origPx: IntArray, maskPx: IntArray, w: Int, h: Int, bounds: Rect,
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
                (Color.blue(c) + noise).toInt().coerceIn(0, 255),
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
    // Context-window inference
    // ---------------------------------------------------------------------

    /**
     * Fills the hole with a single run of the graph over one window that holds
     * the whole hole plus a ring of surrounding context.
     *
     * The graph has a fixed 512x512 field of view. Crops that each cover only
     * part of a large hole give it a mostly-masked view, and it invents content
     * instead of reading the scene - which is what left a dark, blurry smudge
     * behind a removed object. Squeezing the entire hole into one view keeps the
     * mask a small fraction of the model's input, so it reconstructs the real
     * structure (pavement joints continue, grass stays grass) even though the
     * fill is synthesized at a lower resolution than the photo.
     */
    private fun inpaintWithContext(
        origPx: IntArray, maskPx: IntArray, w: Int, h: Int, bounds: Rect
    ): Bitmap {
        val started = System.currentTimeMillis()

        val padX = max(32, bounds.width() / 16)
        val padY = max(32, bounds.height() / 16)
        val x0 = max(0, bounds.left - padX)
        val y0 = max(0, bounds.top - padY)
        val x1 = min(w, bounds.right + padX)
        val y1 = min(h, bounds.bottom + padY)
        val cw = x1 - x0
        val ch = y1 - y0

        val filled = runLaMaOnWindow(origPx, maskPx, w, h, x0, y0, cw, ch)
        val filledPx = IntArray(cw * ch)
        filled.getPixels(filledPx, 0, cw, 0, 0, cw, ch)
        filled.recycle()

        // Composite: the original everywhere, the model output inside the mask.
        val outFull = IntArray(w * h)
        System.arraycopy(origPx, 0, outFull, 0, w * h)
        var painted = 0
        for (j in 0 until ch) {
            val row = (y0 + j) * w
            val fillRow = j * cw
            for (i in 0 until cw) {
                val idx = row + x0 + i
                if (!isMaskPixel(maskPx[idx])) continue
                outFull[idx] = filledPx[fillRow + i]
                painted++
            }
        }
        val side = max(cw, ch)
        Log.d(
            "LaMa",
            "context window ${cw}x${ch} at $x0,$y0 (model scale ${
                min(1f, modelWidth.toFloat() / side)
            }), painted $painted px in ${System.currentTimeMillis() - started} ms"
        )
        return bitmapFromPixels(outFull, w, h)
    }

    /**
     * Runs the graph once over a rectangular crop of the photo and returns a
     * crop-sized fill. The crop is reflection-padded to a square and scaled into
     * the fixed 512x512 input, then the output is mapped back onto the crop; a
     * crop that already fits the input runs at 1:1.
     */
    private fun runLaMaOnWindow(
        origPx: IntArray, maskPx: IntArray, w: Int, h: Int,
        x0: Int, y0: Int, cw: Int, ch: Int
    ): Bitmap {
        val cropImg = bitmapFromPixels(windowPixels(origPx, w, h, x0, y0, cw, ch), cw, ch)
        val cropMask = bitmapFromPixels(windowPixels(maskPx, w, h, x0, y0, cw, ch), cw, ch)

        val (squareImg, squareMask, padL, padT, _, _, squareSize) =
            padToSquareReflect(cropImg, cropMask)
        cropImg.recycle()
        cropMask.recycle()

        val inputImg = if (squareSize == modelWidth) squareImg
        else squareImg.scale(modelWidth, modelHeight, filter = true)
        val inputMask = if (squareSize == modelWidth) squareMask
        else squareMask.scale(modelWidth, modelHeight, filter = true)
        if (inputImg !== squareImg) squareImg.recycle()
        if (inputMask !== squareMask) squareMask.recycle()

        val raw = runLaMaCore(inputImg, inputMask)
        inputImg.recycle()
        inputMask.recycle()

        // Undo the scale and the square padding to land back on the crop.
        val s = modelWidth.toFloat() / squareSize
        val cropL = (padL * s).roundToInt().coerceIn(0, modelWidth - 1)
        val cropT = (padT * s).roundToInt().coerceIn(0, modelHeight - 1)
        val cropR = (cropL + cw * s).roundToInt().coerceIn(cropL + 1, modelWidth)
        val cropB = (cropT + ch * s).roundToInt().coerceIn(cropT + 1, modelHeight)
        val outW = cropR - cropL
        val outH = cropB - cropT
        val cropped = if (cropL == 0 && cropT == 0 && outW == modelWidth && outH == modelHeight) {
            raw
        } else {
            Bitmap.createBitmap(raw, cropL, cropT, outW, outH).also { raw.recycle() }
        }
        return if (cropped.width == cw && cropped.height == ch) cropped
        else cropped.scale(cw, ch, filter = true).also { cropped.recycle() }
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
            "runLaMaCore expects a ${t}x$t tile"
        }
        val total = t * t

        val px = IntArray(total)
        input.getPixels(px, 0, t, 0, 0, t, t)
        val mpx = IntArray(total)
        inputMask.getPixels(mpx, 0, t, 0, 0, t, t)

        val imgArr = FloatArray(total * 3)
        val maskArr = FloatArray(total)
        for (i in 0 until total) {
            val c = px[i]
            imgArr[i] = Color.red(c) / 255f
            imgArr[total + i] = Color.green(c) / 255f
            imgArr[total * 2 + i] = Color.blue(c) / 255f
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

    private fun buildCompositeAlphaMask(mask: Bitmap): Bitmap {
        val blurRadius = 12f
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
