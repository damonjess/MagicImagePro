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
import kotlin.math.absoluteValue
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

        val (dilatedImage0, dilatedMask0) = dilateMaskForCoverage(safeImage, safeMask)

        val (edgeFixedImage, edgeFixedMask) = continueStraightEdgesThroughMask(dilatedImage0, dilatedMask0, w, h)
        dilatedImage0.recycle()
        dilatedMask0.recycle()

        val dilatedImage = edgeFixedImage
        val dilatedMask = edgeFixedMask
        val dilMaskPx = IntArray(w * h)
        dilatedMask.getPixels(dilMaskPx, 0, w, 0, 0, w, h)

        val roi = computeMaskRoi(dilMaskPx, w, h, contextPadFrac = 0.55f)

        val roiW = roi.right - roi.left
        val roiH = roi.bottom - roi.top
        val roiArea = roiW * roiH
        val fullArea = w * h

        val roiIsBigEnough = roiArea >= fullArea * 0.25f

        val (aiFullImage, aiFullQuality) = if (roiIsBigEnough) {
            runFullImageInference(dilatedImage, dilatedMask, w, h) to 0.45f
        } else {
            val roiCropImg = Bitmap.createBitmap(dilatedImage, roi.left, roi.top, roiW, roiH)
            val roiCropMask = Bitmap.createBitmap(dilatedMask, roi.left, roi.top, roiW, roiH)
            val roiInpainted = runRoiInference(roiCropImg, roiCropMask, roiW, roiH)
            roiCropImg.recycle()
            roiCropMask.recycle()

            val roiPasted = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val cvP = Canvas(roiPasted)
            cvP.drawBitmap(dilatedImage, 0f, 0f, null)
            cvP.drawBitmap(roiInpainted, roi.left.toFloat(), roi.top.toFloat(), null)
            roiInpainted.recycle()
            roiPasted to 0.85f
        }

        val exemplarFill = runExemplarTextureFill(dilatedImage, dilMaskPx, w, h, roi)

        val blendedFill = blendAiAndExemplar(
            aiFullImage, exemplarFill, dilMaskPx, w, h, aiFullQuality
        )
        aiFullImage.recycle()
        exemplarFill.recycle()

        val colorCorrected = matchBorderColors(dilatedImage, blendedFill, dilatedMask)
        blendedFill.recycle()

        val maxDim = max(w, h).toFloat()
        val adaptiveBlur = (maxDim / 110f).coerceIn(10f, 42f)
        val widerBlur = (adaptiveBlur * 2.2f).coerceAtMost(65f)

        val softMaskOuter = blurMask(dilatedMask, blurRadius = widerBlur)
        val softMaskInner = blurMask(dilatedMask, blurRadius = adaptiveBlur * 0.45f)
        val gradientMask = buildGradientTransitionMask(softMaskOuter, softMaskInner)

        val seamlessComposite = seamlessPoissonBlend(dilatedImage, colorCorrected, gradientMask, dilatedMask)
        colorCorrected.recycle()
        softMaskOuter.recycle()
        softMaskInner.recycle()
        gradientMask.recycle()

        val finalSeamCleanup = cleanupSeamLine(dilatedImage, seamlessComposite, dilatedMask, adaptiveBlur)
        seamlessComposite.recycle()

        safeImage.recycle()
        safeMask.recycle()
        dilatedImage.recycle()
        dilatedMask.recycle()

        finalSeamCleanup
    }

    // --- Top-level inference helpers ---

    private fun runFullImageInference(image: Bitmap, mask: Bitmap, w: Int, h: Int): Bitmap {
        val paddedImage = padToSquareReflect(image)
        val paddedMask = padToSquareBlack(mask)
        val scaledImage = Bitmap.createScaledBitmap(paddedImage, modelWidth, modelHeight, true)
        val scaledMask = Bitmap.createScaledBitmap(paddedMask, modelWidth, modelHeight, true)

        val rawAiOutput = runLaMaInference(scaledImage, scaledMask)
        scaledImage.recycle()
        scaledMask.recycle()

        val inpaintedPadded = Bitmap.createScaledBitmap(rawAiOutput, paddedImage.width, paddedImage.height, true)
        rawAiOutput.recycle()
        val inpaintedFull = cropFromSquare(inpaintedPadded, w, h)
        paddedImage.recycle()
        paddedMask.recycle()
        inpaintedPadded.recycle()
        return inpaintedFull
    }

    private fun runRoiInference(roiImg: Bitmap, roiMask: Bitmap, roiW: Int, roiH: Int): Bitmap {
        val paddedImage = padToSquareReflect(roiImg)
        val paddedMask = padToSquareBlack(roiMask)
        val scaledImage = Bitmap.createScaledBitmap(paddedImage, modelWidth, modelHeight, true)
        val scaledMask = Bitmap.createScaledBitmap(paddedMask, modelWidth, modelHeight, true)

        val rawAiOutput = runLaMaInference(scaledImage, scaledMask)
        scaledImage.recycle()
        scaledMask.recycle()

        val targetSize = max(paddedImage.width, paddedImage.height)
        val inpaintedPadded = Bitmap.createScaledBitmap(rawAiOutput, targetSize, targetSize, true)
        rawAiOutput.recycle()
        val roiSquare = cropFromSquare(inpaintedPadded, roiImg.width, roiImg.height)
        paddedImage.recycle()
        paddedMask.recycle()
        inpaintedPadded.recycle()
        return roiSquare
    }

    private fun runLaMaInference(scaledImage: Bitmap, scaledMask: Bitmap): Bitmap {
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
        return convertOutputToBitmap(outputBuffer, modelWidth, modelHeight, outIsNCHW)
    }

    // --- ROI / BBox helpers ---

    private fun computeMaskRoi(maskPx: IntArray, w: Int, h: Int, contextPadFrac: Float): Rect {
        var minX = w
        var minY = h
        var maxX = 0
        var maxY = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (Color.red(maskPx[y * w + x]) > 127) {
                    if (x < minX) minX = x
                    if (y < minY) minY = y
                    if (x > maxX) maxX = x
                    if (y > maxY) maxY = y
                }
            }
        }
        if (maxX < minX || maxY < minY) {
            return Rect(0, 0, w, h)
        }
        val bw = maxX - minX + 1
        val bh = maxY - minY + 1
        val padX = (bw * contextPadFrac).roundToInt()
        val padY = (bh * contextPadFrac).roundToInt()
        val l = max(0, minX - padX)
        val t = max(0, minY - padY)
        val r = min(w, maxX + padX + 1)
        val b = min(h, maxY + padY + 1)
        val rw = r - l
        val rh = b - t
        val squareSide = max(rw, rh)
        val centerX = (l + r) / 2
        val centerY = (t + b) / 2
        var nl = max(0, centerX - squareSide / 2)
        var nr = min(w, nl + squareSide)
        nl = max(0, nr - squareSide)
        var nt = max(0, centerY - squareSide / 2)
        var nb = min(h, nt + squareSide)
        nt = max(0, nb - squareSide)
        return Rect(nl, nt, nr, nb)
    }

    // --- Multi-texture exemplar inpainting (fixes grass/patio boundary failure) ---

    private fun runExemplarTextureFill(
        original: Bitmap,
        maskPx: IntArray,
        w: Int,
        h: Int,
        roi: Rect
    ): Bitmap {
        val origPx = IntArray(w * h)
        original.getPixels(origPx, 0, w, 0, 0, w, h)

        val outPx = IntArray(w * h)
        System.arraycopy(origPx, 0, outPx, 0, w * h)

        val sampleBand = (max(w, h) / 60f).coerceIn(6f, 24f).toInt()

        data class Sample(val color: Int, val x: Int, val y: Int)
        val outsideSamples = mutableListOf<Sample>()

        val searchL = max(0, roi.left - sampleBand * 3)
        val searchT = max(0, roi.top - sampleBand * 3)
        val searchR = min(w, roi.right + sampleBand * 3)
        val searchB = min(h, roi.bottom + sampleBand * 3)
        val step = max(1, ((searchR - searchL) * (searchB - searchT)) / 25000)
        for (y in searchT until searchB step step) {
            for (x in searchL until searchR step step) {
                val idx = y * w + x
                if (Color.red(maskPx[idx]) <= 127) {
                    var nearMask = false
                    val dyLim = sampleBand
                    val dxLim = sampleBand
                    checkBorder@ for (dy in -dyLim..dyLim) {
                        for (dx in -dxLim..dxLim) {
                            val nx = x + dx
                            val ny = y + dy
                            if (nx in 0 until w && ny in 0 until h) {
                                if (Color.red(maskPx[ny * w + nx]) > 127) {
                                    nearMask = true
                                    break@checkBorder
                                }
                            }
                        }
                    }
                    if (nearMask) outsideSamples.add(Sample(origPx[idx], x, y))
                }
            }
        }

        if (outsideSamples.size < 8) {
            val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            result.setPixels(outPx, 0, w, 0, 0, w, h)
            return result
        }

        fun labDist(a: Int, b: Int): Float {
            val ar = Color.red(a).toFloat(); val ag = Color.green(a).toFloat(); val ab = Color.blue(a).toFloat()
            val br = Color.red(b).toFloat(); val bg = Color.green(b).toFloat(); val bb = Color.blue(b).toFloat()
            val dr = ar - br; val dg = ag - bg; val db = ab - bb
            return sqrt(dr * dr + dg * dg + db * db)
        }

        val fillMap = IntArray(w * h) { -1 }
        val fillConfidence = FloatArray(w * h) { Float.MAX_VALUE }

        val roiL = roi.left; val roiT = roi.top; val roiR = roi.right; val roiB = roi.bottom

        val maxTextureK = 9.coerceAtMost(outsideSamples.size)

        val passOrder = mutableListOf<Pair<Int, Int>>()
        for (y in roiT until roiB) for (x in roiL until roiR) {
            if (Color.red(maskPx[y * w + x]) > 127) passOrder.add(x to y)
        }
        passOrder.sortBy { (x, y) ->
            var d = Int.MAX_VALUE
            for (s in outsideSamples) {
                val dd = (s.x - x) * (s.x - x) + (s.y - y) * (s.y - y)
                if (dd < d) d = dd
            }
            d
        }

        for ((x, y) in passOrder) {
            val idx = y * w + x
            val ctxR = (sampleBand * 0.6f).toInt().coerceAtLeast(2)

            val neighbors = mutableListOf<Int>()
            for (dy in -ctxR..ctxR step max(1, ctxR / 3)) {
                for (dx in -ctxR..ctxR step max(1, ctxR / 3)) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx in 0 until w && ny in 0 until h) {
                        val nIdx = ny * w + nx
                        val color = if (fillMap[nIdx] != -1) origPx[fillMap[nIdx]]
                        else if (Color.red(maskPx[nIdx]) <= 127) origPx[nIdx]
                        else continue
                        neighbors.add(color)
                    }
                }
            }

            var nR = 0f; var nG = 0f; var nB = 0f
            for (n in neighbors) { nR += Color.red(n); nG += Color.green(n); nB += Color.blue(n) }
            val nc = neighbors.size
            if (nc > 0) { nR /= nc; nG /= nc; nB /= nc }
            val neighborAvg = Color.rgb(nR.coerceIn(0f,255f).toInt(), nG.coerceIn(0f,255f).toInt(), nB.coerceIn(0f,255f).toInt())

            var bestIdx = -1
            var bestScore = Float.MAX_VALUE
            for ((si, s) in outsideSamples.withIndex()) {
                val distSq = ((s.x - x) * (s.x - x) + (s.y - y) * (s.y - y)).toFloat()
                val spatial = sqrt(distSq) / (sampleBand * 8f).coerceAtLeast(1f)
                val colorD = labDist(origPx[s.x + s.y * w], neighborAvg) / 60f.coerceAtLeast(1f)
                val score = colorD * 1.05f + spatial * 0.55f
                if (score < bestScore) { bestScore = score; bestIdx = si }
            }

            if (bestIdx != -1) {
                val s = outsideSamples[bestIdx]
                val sx0 = s.x - (x - roiL)
                val sy0 = s.y - (y - roiT)
                val sx1 = sx0 + (roiR - roiL) - 1
                val sy1 = sy0 + (roiB - roiT) - 1

                val sampleRoiOk = sx0 >= 0 && sy0 >= 0 && sx1 < w && sy1 < h
                val tIdx = if (sampleRoiOk) {
                    val localX = x - roiL
                    val localY = y - roiT
                    val srcX = sx0 + localX
                    val srcY = sy0 + localY
                    if (srcX in 0 until w && srcY in 0 until h &&
                        Color.red(maskPx[srcY * w + srcX]) <= 127) {
                        srcY * w + srcX
                    } else s.y * w + s.x
                } else s.y * w + s.x

                fillMap[idx] = tIdx
                fillConfidence[idx] = bestScore
            }
        }

        for (y in roiT until roiB) {
            for (x in roiL until roiR) {
                val idx = y * w + x
                if (Color.red(maskPx[idx]) > 127 && fillMap[idx] != -1) {
                    val r = 3
                    var sr = 0f; var sg = 0f; var sb = 0f; var wt = 0f
                    for (dy in -r..r) {
                        for (dx in -r..r) {
                            val nx = x + dx
                            val ny = y + dy
                            if (nx in 0 until w && ny in 0 until h) {
                                val nIdx = ny * w + nx
                                val src = if (fillMap[nIdx] != -1) origPx[fillMap[nIdx]] else origPx[nIdx]
                                val d = sqrt((dx * dx + dy * dy).toFloat())
                                val wd = 1f / (1f + d * 0.5f)
                                sr += Color.red(src) * wd
                                sg += Color.green(src) * wd
                                sb += Color.blue(src) * wd
                                wt += wd
                            }
                        }
                    }
                    outPx[idx] = Color.rgb(
                        (sr / wt).coerceIn(0f,255f).toInt(),
                        (sg / wt).coerceIn(0f,255f).toInt(),
                        (sb / wt).coerceIn(0f,255f).toInt()
                    )
                }
            }
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(outPx, 0, w, 0, 0, w, h)
        return result
    }

    private fun blendAiAndExemplar(
        aiImg: Bitmap,
        exemplarImg: Bitmap,
        maskPx: IntArray,
        w: Int,
        h: Int,
        aiWeightBase: Float
    ): Bitmap {
        val aiPx = IntArray(w * h)
        val exPx = IntArray(w * h)
        aiImg.getPixels(aiPx, 0, w, 0, 0, w, h)
        exemplarImg.getPixels(exPx, 0, w, 0, 0, w, h)
        val outPx = IntArray(w * h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                if (Color.red(maskPx[idx]) <= 127) {
                    outPx[idx] = aiPx[idx]
                    continue
                }
                val borderDist = distanceToMaskBorder(x, y, maskPx, w, h)
                val bd = borderDist.coerceIn(0f, 50f)
                val edgeBias = (1f - bd / 50f)
                val exemplarW = 0.55f + 0.35f * edgeBias
                val aiW = 1f - exemplarW

                val a = aiPx[idx]
                val e = exPx[idx]
                val r = (Color.red(a) * aiW + Color.red(e) * exemplarW).toInt().coerceIn(0, 255)
                val g = (Color.green(a) * aiW + Color.green(e) * exemplarW).toInt().coerceIn(0, 255)
                val b = (Color.blue(a) * aiW + Color.blue(e) * exemplarW).toInt().coerceIn(0, 255)
                outPx[idx] = Color.rgb(r, g, b)
            }
        }
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(outPx, 0, w, 0, 0, w, h)
        return result
    }

    // --- Helper Methods ---

    private fun continueStraightEdgesThroughMask(
        image: Bitmap,
        mask: Bitmap,
        w: Int,
        h: Int
    ): Pair<Bitmap, Bitmap> {
        val origPx = IntArray(w * h)
        val maskPx = IntArray(w * h)
        image.getPixels(origPx, 0, w, 0, 0, w, h)
        mask.getPixels(maskPx, 0, w, 0, 0, w, h)

        val minImgDim = min(w, h)
        val gradThresh = (minImgDim / 160f).coerceIn(10f, 28f)
        val minLineLen = (minImgDim / 9f).coerceIn(40f, 200f).toInt()
        val edgeBand = (minImgDim / 80f).coerceIn(4f, 16f).toInt()

        data class EdgeSample(
            val x1: Int, val y1: Int, val x2: Int, val y2: Int,
            val angle: Float,
            val lineColorL: Float, val lineColorA: Float, val lineColorB: Float,
            val sideColorL: Float, val sideColorA: Float, val sideColorB: Float,
            val perpSide: Int,
            val strength: Float
        )

        fun rgb2Lab(c: Int): Triple<Float, Float, Float> {
            var r = Color.red(c) / 255.0; var g = Color.green(c) / 255.0; var b = Color.blue(c) / 255.0
            r = if (r > 0.04045) Math.pow((r + 0.055) / 1.055, 2.4) else r / 12.92
            g = if (g > 0.04045) Math.pow((g + 0.055) / 1.055, 2.4) else g / 12.92
            b = if (b > 0.04045) Math.pow((b + 0.055) / 1.055, 2.4) else b / 12.92
            val x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
            val y = (r * 0.2126 + g * 0.7152 + b * 0.0722)
            val z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
            val fx = if (x > 0.008856) Math.cbrt(x) else (7.787 * x + 16.0 / 116.0)
            val fy = if (y > 0.008856) Math.cbrt(y) else (7.787 * y + 16.0 / 116.0)
            val fz = if (z > 0.008856) Math.cbrt(z) else (7.787 * z + 16.0 / 116.0)
            return Triple((116.0 * fy - 16.0).toFloat(), (500.0 * (fx - fy)).toFloat(), (200.0 * (fy - fz)).toFloat())
        }

        val labBuf = Array(h) { Array(w) { Triple(0f, 0f, 0f) } }
        val lumImg = FloatArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val lab = rgb2Lab(origPx[y * w + x])
                labBuf[y][x] = lab
                lumImg[y * w + x] = lab.first
            }
        }

        val edges = mutableListOf<EdgeSample>()
        val sampleStep = max(1, min(w, h) / 600)

        for (y in edgeBand until h - edgeBand step sampleStep) {
            for (x in edgeBand until w - edgeBand step sampleStep) {
                val idx = y * w + x
                if (Color.red(maskPx[idx]) > 127) continue

                var nearMask = false
                checkNear@ for (dy in -edgeBand..edgeBand) {
                    for (dx in -edgeBand..edgeBand) {
                        val nx = x + dx; val ny = y + dy
                        if (nx in 0 until w && ny in 0 until h && Color.red(maskPx[ny * w + nx]) > 127) {
                            nearMask = true; break@checkNear
                        }
                    }
                }
                if (!nearMask) continue

                val lc = lumImg[idx]
                for (o in 0 until 8) {
                    val ang = o * Math.PI / 8.0
                    val dx = Math.cos(ang).toFloat()
                    val dy = Math.sin(ang).toFloat()
                    val pdx = -dy; val pdy = dx

                    var fwd = 0
                    var fwLum = lc
                    while (fwd < minLineLen) {
                        val nx = (x + dx * fwd).roundToInt()
                        val ny = (y + dy * fwd).roundToInt()
                        if (nx !in edgeBand until w - edgeBand || ny !in edgeBand until h - edgeBand) break
                        if (Color.red(maskPx[ny * w + nx]) > 127) break
                        fwLum = lumImg[ny * w + nx]
                        fwd++
                    }
                    var bwd = 0
                    var bwLum = lc
                    while (bwd < minLineLen) {
                        val nx = (x - dx * bwd).roundToInt()
                        val ny = (y - dy * bwd).roundToInt()
                        if (nx !in edgeBand until w - edgeBand || ny !in edgeBand until h - edgeBand) break
                        if (Color.red(maskPx[ny * w + nx]) > 127) break
                        bwLum = lumImg[ny * w + nx]
                        bwd++
                    }

                    val totLen = fwd + bwd
                    if (totLen < minLineLen) continue

                    val linePx1 = ((x - dx * (bwd - 1)).roundToInt()) to ((y - dy * (bwd - 1)).roundToInt())
                    val linePx2 = ((x + dx * (fwd - 1)).roundToInt()) to ((y + dy * (fwd - 1)).roundToInt())

                    var perpSum1 = 0f
                    var perpSum2 = 0f
                    var pc = 0
                    for (t in -totLen / 2..totLen / 2 step max(1, totLen / 10)) {
                        val lx = x + (dx * t).toInt()
                        val ly = y + (dy * t).toInt()
                        if (lx !in (edgeBand + 2) until w - edgeBand - 2 || ly !in (edgeBand + 2) until h - edgeBand - 2) continue
                        val nx1 = (lx + pdx * 2).toInt(); val ny1 = (ly + pdy * 2).toInt()
                        val nx2 = (lx - pdx * 2).toInt(); val ny2 = (ly - pdy * 2).toInt()
                        if (nx1 in 0 until w && ny1 in 0 until h && Color.red(maskPx[ny1 * w + nx1]) <= 127) perpSum1 += lumImg[ny1 * w + nx1]
                        if (nx2 in 0 until w && ny2 in 0 until h && Color.red(maskPx[ny2 * w + nx2]) <= 127) perpSum2 += lumImg[ny2 * w + nx2]
                        pc++
                    }
                    if (pc == 0) continue
                    perpSum1 /= pc; perpSum2 /= pc
                    val d1 = Math.abs(perpSum1 - (fwLum + bwLum) / 2f)
                    val d2 = Math.abs(perpSum2 - (fwLum + bwLum) / 2f)
                    val contrast = max(d1, d2)
                    if (contrast < gradThresh) continue

                    val sideIdx = if (d1 >= d2) 1 else -1
                    var lineSL = 0f; var lineSA = 0f; var lineSB = 0f; var lineCt = 0f
                    var sideSL = 0f; var sideSA = 0f; var sideSB = 0f; var sideCt = 0f
                    for (t in -totLen / 2..totLen / 2 step max(1, totLen / 15)) {
                        val lx = x + (dx * t).toInt()
                        val ly = y + (dy * t).toInt()
                        if (lx in edgeBand until w - edgeBand && ly in edgeBand until h - edgeBand) {
                            if (Color.red(maskPx[ly * w + lx]) <= 127) {
                                val l = labBuf[ly][lx]
                                lineSL += l.first; lineSA += l.second; lineSB += l.third; lineCt += 1f
                            }
                            val sx = (lx + pdx * 3 * sideIdx).toInt()
                            val sy = (ly + pdy * 3 * sideIdx).toInt()
                            if (sx in 0 until w && sy in 0 until h && Color.red(maskPx[sy * w + sx]) <= 127) {
                                val l = labBuf[sy][sx]
                                sideSL += l.first; sideSA += l.second; sideSB += l.third; sideCt += 1f
                            }
                        }
                    }
                    if (lineCt < 3 || sideCt < 3) continue

                    edges.add(
                        EdgeSample(
                            x1 = linePx1.first, y1 = linePx1.second,
                            x2 = linePx2.first, y2 = linePx2.second,
                            angle = ang.toFloat(),
                            lineColorL = lineSL / lineCt,
                            lineColorA = lineSA / lineCt,
                            lineColorB = lineSB / lineCt,
                            sideColorL = sideSL / sideCt,
                            sideColorA = sideSA / sideCt,
                            sideColorB = sideSB / sideCt,
                            perpSide = sideIdx,
                            strength = contrast
                        )
                    )
                }
            }
        }

        if (edges.size < 2) {
            val outI = image.copy(Bitmap.Config.ARGB_8888, true)
            val outM = mask.copy(Bitmap.Config.ARGB_8888, true)
            return Pair(outI, outM)
        }

        val binnedAngles = IntArray(16)
        for (e in edges) {
            val a = (((e.angle + Math.PI.toFloat()) % Math.PI.toFloat()) / Math.PI.toFloat() * 16).toInt().coerceIn(0, 15)
            binnedAngles[a]++
        }
        var domBin = 0; var domCount = 0
        for (i in 0 until 16) if (binnedAngles[i] > domCount) { domCount = binnedAngles[i]; domBin = i }
        val dominantAng = (domBin / 16.0 * Math.PI - Math.PI / 2).toFloat()
        val angTol = (Math.PI / 12.0).toFloat()

        val matchedEdges = edges.filter { e ->
            val da = Math.abs(((e.angle - dominantAng + Math.PI.toFloat() * 3) % (Math.PI.toFloat() * 2)) - Math.PI.toFloat())
            da < angTol || Math.abs(da - Math.PI.toFloat()) < angTol
        }.sortedByDescending { it.strength }.take(24)

        if (matchedEdges.isEmpty()) {
            val outI = image.copy(Bitmap.Config.ARGB_8888, true)
            val outM = mask.copy(Bitmap.Config.ARGB_8888, true)
            return Pair(outI, outM)
        }

        fun lab2Rgb(l: Float, a: Float, b: Float): Int {
            val fy = (l + 16f) / 116f
            val fx = a / 500f + fy
            val fz = fy - b / 200f
            val fy3 = fy * fy * fy; val fx3 = fx * fx * fx; val fz3 = fz * fz * fz
            val yr = if (fy3 > 0.008856) fy3 else (fy - 16f / 116f) / 7.787f
            val xr = if (fx3 > 0.008856) fx3 else (fx - 16f / 116f) / 7.787f
            val zr = if (fz3 > 0.008856) fz3 else (fz - 16f / 116f) / 7.787f
            val X = xr * 0.95047f; val Y = yr; val Z = zr * 1.08883f
            var R = X * 3.2406f + Y * -1.5372f + Z * -0.4986f
            var G = X * -0.9689f + Y * 1.8758f + Z * 0.0415f
            var B = X * 0.0557f + Y * -0.2040f + Z * 1.0570f
            R = if (R > 0.0031308f) (1.055f * Math.pow(R.toDouble(), 1.0 / 2.4).toFloat() - 0.055f) else 12.92f * R
            G = if (G > 0.0031308f) (1.055f * Math.pow(G.toDouble(), 1.0 / 2.4).toFloat() - 0.055f) else 12.92f * G
            B = if (B > 0.0031308f) (1.055f * Math.pow(B.toDouble(), 1.0 / 2.4).toFloat() - 0.055f) else 12.92f * B
            return Color.rgb(R.coerceIn(0f, 1f) * 255, G.coerceIn(0f, 1f) * 255, B.coerceIn(0f, 1f) * 255)
        }

        val affectedPx = IntArray(w * h) { -1 }
        val paintRadius = edgeBand.coerceAtLeast(3)

        for (e in matchedEdges) {
            val ang = dominantAng
            val dx = Math.cos(ang.toDouble()).toFloat()
            val dy = Math.sin(ang.toDouble()).toFloat()
            val pdx = -dy
            val pdy = dx
            val ref = e.y1 * w + e.x1
            val refL = lumImg[ref]

            var bestT = 0.0
            var bestTScore = -1e9f
            var steps = 0
            while (steps < 200) {
                val t = bestT + (steps - 100) * 0.5
                val mx = ((e.x1 + e.x2) / 2.0 + dx * t).roundToInt()
                val my = ((e.y1 + e.y2) / 2.0 + dy * t).roundToInt()
                if (mx !in 0 until w || my !in 0 until h) { steps++; continue }
                val mid = my * w + mx
                if (Color.red(maskPx[mid]) <= 127) { steps++; continue }
                var s = 0f
                val n = 10
                for (k in -n..n) {
                    val px = (mx + pdx * k).toInt(); val py = (my + pdy * k).toInt()
                    if (px in 0 until w && py in 0 until h) {
                        if (Color.red(maskPx[py * w + px]) <= 127) {
                            s += 1f / (1f + Math.abs(refL - lumImg[py * w + px]) / 15f)
                        }
                    }
                }
                if (s > bestTScore) { bestTScore = s; bestT = t }
                steps++
            }

            val midx = ((e.x1 + e.x2) / 2.0 + dx * bestT).toFloat()
            val midy = ((e.y1 + e.y2) / 2.0 + dy * bestT).toFloat()

            val len1 = Math.hypot((e.x1 - midx).toDouble(), (e.y1 - midy).toDouble()).toFloat()
            val len2 = Math.hypot((e.x2 - midx).toDouble(), (e.y2 - midy).toDouble()).toFloat()
            val extendLen = max(len1, len2) + paintRadius * 2

            for (t in (-extendLen).toInt()..extendLen.toInt()) {
                val lx = (midx + dx * t).roundToInt()
                val ly = (midy + dy * t).roundToInt()
                if (lx !in paintRadius until w - paintRadius || ly !in paintRadius until h - paintRadius) continue

                for (pr in -paintRadius..paintRadius) {
                    val px = (lx + pdx * pr).toInt()
                    val py = (ly + pdy * pr).toInt()
                    if (px in 0 until w && py in 0 until h && Color.red(maskPx[py * w + px]) > 127) {
                        val distLine = Math.abs(pr) / paintRadius.toFloat()
                        val useL: Float; val useA: Float; val useB: Float
                        if (pr * e.perpSide <= 0) {
                            val k = (1f - distLine).coerceIn(0f, 1f)
                            useL = e.lineColorL * k + e.sideColorL * (1f - k)
                            useA = e.lineColorA * k + e.sideColorA * (1f - k)
                            useB = e.lineColorB * k + e.sideColorB * (1f - k)
                        } else {
                            val k = distLine.coerceIn(0f, 1f)
                            useL = e.sideColorL * (0.65f + 0.35f * k) + e.lineColorL * (0.35f - 0.35f * k)
                            useA = e.sideColorA * (0.65f + 0.35f * k) + e.lineColorA * (0.35f - 0.35f * k)
                            useB = e.sideColorB * (0.65f + 0.35f * k) + e.lineColorB * (0.35f - 0.35f * k)
                        }
                        val existing = affectedPx[py * w + px]
                        val newColor = lab2Rgb(useL, useA, useB)
                        if (existing == -1) {
                            affectedPx[py * w + px] = newColor
                        } else {
                            val er = Color.red(existing); val eg = Color.green(existing); val eb = Color.blue(existing)
                            val nr = Color.red(newColor); val ng = Color.green(newColor); val nb = Color.blue(newColor)
                            affectedPx[py * w + px] = Color.rgb(
                                ((er + nr) / 2), ((eg + ng) / 2), ((eb + nb) / 2)
                            )
                        }
                    }
                }
            }
        }

        val outPx = IntArray(w * h)
        System.arraycopy(origPx, 0, outPx, 0, w * h)
        for (i in 0 until w * h) {
            if (affectedPx[i] != -1) outPx[i] = affectedPx[i]
        }

        val newMaskPx = IntArray(w * h)
        System.arraycopy(maskPx, 0, newMaskPx, 0, w * h)
        var affectedCount = 0
        for (i in 0 until w * h) {
            if (affectedPx[i] != -1 && Color.red(newMaskPx[i]) > 127) {
                newMaskPx[i] = Color.BLACK
                affectedCount++
            }
        }

        val outI = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        outI.setPixels(outPx, 0, w, 0, 0, w, h)
        val outM = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        outM.setPixels(newMaskPx, 0, w, 0, 0, w, h)
        return Pair(outI, outM)
    }

    private fun dilateMaskForCoverage(image: Bitmap, mask: Bitmap): Pair<Bitmap, Bitmap> {
        val outImage = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        val outMask = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)

        val w = image.width
        val h = image.height
        val maxDim = max(w, h)
        val radiusPx = (maxDim / 190f).coerceIn(5f, 20f).toInt()

        val maskPixels = IntArray(w * h)
        mask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val temp = IntArray(w * h)

        var passes = 0
        var r = radiusPx
        while (r > 0 && passes < 6) {
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

        val edgeDilate = (maxDim / 520f).coerceIn(2f, 5f).toInt()
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
            val blend = (o * 0.65f + inn * 0.35f)
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

        val bandWidth = (min(w, h) / 45f).coerceIn(5f, 22f).toInt()

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
                        val t = (borderDist / 60f).coerceIn(0f, 1f)
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
        val maxSearch = 80
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

        val seamWidth = (blurAmount * 0.85f).coerceIn(4f, 24f)

        val seamMaskPixels = IntArray(w * h)
        val maskPx = IntArray(w * h)
        mask.getPixels(maskPx, 0, w, 0, 0, w, h)

        val maxDim = max(w, h)
        val searchR = (maxDim / 90f).coerceIn(3f, 16f).toInt()

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
                    for (d in 1..(seamWidth * 3f).toInt()) {
                        if (insideDist == Int.MAX_VALUE) {
                            searchBorder@ for (dy in -d..d) {
                                for (dx in -d..d) {
                                    if (dx.absoluteValue != d && dy.absoluteValue != d) continue
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
                                    if (dx.absoluteValue != d && dy.absoluteValue != d) continue
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
                        val boosted = (falloff * 1.25f).coerceIn(0f, 1f)
                        val a = (boosted * 255f).coerceIn(0f, 255f).toInt()
                        seamMaskPixels[idx] = Color.argb(a, 255, 255, 255)
                    }
                }
            }
        }

        val seamMaskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        seamMaskBitmap.setPixels(seamMaskPixels, 0, w, 0, 0, w, h)

        val blurredSeamMask = blurMaskGeneric(seamMaskBitmap, (seamWidth * 0.85f).coerceIn(4f, 25f))

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

                val r = 4
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
                            val wd = 1f / (1f + d * 0.45f)
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

                val seamStrength = seamAlpha.coerceAtMost(0.9f)
                val mr = cr * (1f - seamStrength) + avgR * seamStrength
                val mg = cg * (1f - seamStrength) + avgG * seamStrength
                val mb = cb * (1f - seamStrength) + avgB * seamStrength

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
        val clampedRadius = blurRadius.coerceIn(1f, 25f)
        if (blurRadius <= 25f) {
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
            paint.maskFilter = BlurMaskFilter(clampedRadius, BlurMaskFilter.Blur.NORMAL)

            val alphaMask = alphaSource.extractAlpha(paint, null)

            val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG)
            drawPaint.color = Color.WHITE
            canvas.drawBitmap(alphaMask, 0f, 0f, drawPaint)

            alphaMask.recycle()
            alphaSource.recycle()
            return outputBitmap
        } else {
            var accum = mask.copy(Bitmap.Config.ARGB_8888, true)
            var remaining = blurRadius
            while (remaining > 0f) {
                val step = remaining.coerceAtMost(25f)
                val next = blurMaskOnePass(accum, step)
                accum.recycle()
                accum = next
                remaining -= step
            }
            return accum
        }
    }

    private fun blurMaskOnePass(mask: Bitmap, radius: Float): Bitmap {
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
        paint.maskFilter = BlurMaskFilter(radius.coerceIn(1f, 25f), BlurMaskFilter.Blur.NORMAL)

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
