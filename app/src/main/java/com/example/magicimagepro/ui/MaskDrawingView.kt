package com.example.magicimagepro.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import androidx.core.view.doOnLayout
import kotlin.math.hypot

enum class ToolMode { BRUSH, LASSO, ERASER }

// Stores a single stroke for Undo/Redo history
data class DrawAction(val path: Path, val mode: ToolMode, val size: Float)

class MaskDrawingView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    
    enum class EditMode { BRUSH, SMART_SELECT }
    var editMode = EditMode.BRUSH

    interface OnTapListener {
        fun onTap(x: Float, y: Float)
    }
    var onTapListener: OnTapListener? = null

    /**
     * Notifies the host whenever the photo/mask view transform changes (pinch
     * zoom & pan), so sibling views showing the same photo can follow along.
     * Args: zoom, panX, panY where screen = pan + zoom * fitPoint.
     */
    var onTransformListener: ((zoom: Float, panX: Float, panY: Float) -> Unit)? = null

    private var imageBitmap: Bitmap? = null
    private var baseMaskBitmap: Bitmap? = null
    var currentMode = ToolMode.BRUSH
    var brushSize = 60f
    var cursorOffset = 150f // Pushes the brush up from your finger
    
    // History Stacks
    private val actionStack = mutableListOf<DrawAction>()
    private val redoStack = mutableListOf<DrawAction>()
    private var currentPath = Path()
    
    // Core Canvases
    private var displayMaskBitmap: Bitmap? = null
    private var displayMaskCanvas: Canvas? = null
    private var exportMaskBitmap: Bitmap? = null
    private var exportMaskCanvas: Canvas? = null
    
    // Paints
    private val maskColor = Color.parseColor("#993DDC84") // Semi-transparent Green
    
    private val basePaint = Paint().apply {
        isAntiAlias = true
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    
    private val cursorPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    // Loupe (Magnifier) Paints
    private val crosshairPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val loupeBorderPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 8f
        isAntiAlias = true
    }
    
    // Live tracking
    private var isDrawing = false
    private var touchX = -1f
    private var touchY = -1f
    
    // Layout Math
    private var scaleFactor = 1f
    private var offsetX = 0f
    private var offsetY = 0f
    private val displayRect = RectF()

    // View-level zoom/pan applied on top of the fit-center rect. Two fingers
    // pinch-zoom and pan; one finger draws. screen = pan + zoom * fitPoint.
    private var viewZoom = 1f
    private var viewPanX = 0f
    private var viewPanY = 0f
    private var isPinching = false
    private var pinchStartSpan = 1f
    private var pinchStartZoom = 1f
    private var pinchStartPanX = 0f
    private var pinchStartPanY = 0f
    private var pinchStartFocalX = 0f
    private var pinchStartFocalY = 0f

    fun setImage(bitmap: Bitmap) {
        imageBitmap = bitmap
        
        // A new photo always starts unzoomed.
        viewZoom = 1f
        viewPanX = 0f
        viewPanY = 0f
        isPinching = false
        notifyTransform()

        baseMaskBitmap?.recycle()
        baseMaskBitmap = null

        displayMaskBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        displayMaskCanvas = Canvas(displayMaskBitmap!!)
        
        exportMaskBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        exportMaskCanvas = Canvas(exportMaskBitmap!!)
        
        actionStack.clear()
        redoStack.clear()
        
        doOnLayout { calculateDisplayRect(); redrawHistory() }
    }
    
    private fun calculateDisplayRect() {
        imageBitmap?.let { bmp ->
            if (width == 0 || height == 0) return
            scaleFactor = minOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)
            val scaledW = bmp.width * scaleFactor
            val scaledH = bmp.height * scaleFactor
            offsetX = (width - scaledW) / 2f
            offsetY = (height - scaledH) / 2f
            displayRect.set(offsetX, offsetY, offsetX + scaledW, offsetY + scaledH)
            clampPan()
            notifyTransform()
        }
    }

    private fun notifyTransform() {
        onTransformListener?.invoke(viewZoom, viewPanX, viewPanY)
    }

    /** Keeps the zoomed image from being dragged completely off the view. */
    private fun clampPan() {
        val bmp = imageBitmap ?: return
        if (viewZoom <= 1f) {
            viewZoom = 1f
            viewPanX = 0f
            viewPanY = 0f
            return
        }
        // Content spans [pan + zoom*offset .. pan + zoom*(offset+size)] per axis.
        val leftEdge = viewPanX + viewZoom * offsetX
        val rightEdge = viewPanX + viewZoom * (offsetX + bmp.width * scaleFactor)
        val topEdge = viewPanY + viewZoom * offsetY
        val bottomEdge = viewPanY + viewZoom * (offsetY + bmp.height * scaleFactor)
        if (rightEdge < 0f) viewPanX -= rightEdge
        if (leftEdge > width) viewPanX -= (leftEdge - width)
        if (bottomEdge < 0f) viewPanY -= bottomEdge
        if (topEdge > height) viewPanY -= (topEdge - height)
    }

    private fun pinchSpan(e: MotionEvent): Float {
        val dx = e.getX(0) - e.getX(1)
        val dy = e.getY(0) - e.getY(1)
        return hypot(dx, dy).coerceAtLeast(1f)
    }

    private fun pinchFocal(e: MotionEvent): Pair<Float, Float> =
        Pair((e.getX(0) + e.getX(1)) / 2f, (e.getY(0) + e.getY(1)) / 2f)

    fun setMask(mask: Bitmap) {
        val bmp = imageBitmap ?: return
        val scaledMask = if (mask.width != bmp.width || mask.height != bmp.height) {
            Bitmap.createScaledBitmap(mask, bmp.width, bmp.height, true)
        } else {
            mask
        }

        actionStack.clear()
        redoStack.clear()

        baseMaskBitmap?.recycle()
        baseMaskBitmap = scaledMask.copy(Bitmap.Config.ARGB_8888, true)
        
        exportMaskBitmap = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
        exportMaskCanvas = Canvas(exportMaskBitmap!!)

        redrawHistory()
    }
    
    fun getMaskBitmap(): Bitmap? {
        val export = exportMaskBitmap ?: return null
        val w = export.width
        val h = export.height
        val cv = exportMaskCanvas ?: return null

        cv.drawColor(Color.BLACK, PorterDuff.Mode.CLEAR)
        cv.drawColor(Color.BLACK)

        baseMaskBitmap?.let { base ->
            cv.drawBitmap(base, 0f, 0f, null)
        }

        val exportPaint = Paint(basePaint).apply {
            color = Color.WHITE
            isFilterBitmap = false
            isAntiAlias = true
        }
        val eraserPaint = Paint(basePaint).apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
            isFilterBitmap = false
            isAntiAlias = true
        }
        val fillPaint = Paint(basePaint).apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            isFilterBitmap = false
            isAntiAlias = true
        }

        for (action in actionStack) {
            val sz = action.size
            exportPaint.strokeWidth = sz
            eraserPaint.strokeWidth = sz

            when (action.mode) {
                ToolMode.BRUSH -> {
                    exportPaint.style = Paint.Style.STROKE
                    exportPaint.strokeCap = Paint.Cap.ROUND
                    exportPaint.strokeJoin = Paint.Join.ROUND
                    cv.drawPath(action.path, exportPaint)
                }
                ToolMode.LASSO -> {
                    cv.drawPath(action.path, fillPaint)
                }
                ToolMode.ERASER -> {
                    eraserPaint.style = Paint.Style.STROKE
                    eraserPaint.strokeCap = Paint.Cap.ROUND
                    eraserPaint.strokeJoin = Paint.Join.ROUND
                    cv.drawPath(action.path, eraserPaint)
                }
            }
        }

        // Binarize by LUMINANCE only, never by alpha. The export canvas is filled with
        // opaque black (alpha = 255), so an alpha check would mark every unselected pixel
        // as selected and send the whole image to the model as one giant hole (-> white
        // output). White strokes => selected (keep as WHITE), black background => not.
        val pixels = IntArray(w * h)
        export.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) {
            val r = Color.red(pixels[i])
            val g = Color.green(pixels[i])
            val b = Color.blue(pixels[i])
            pixels[i] = if (r > 85 || g > 85 || b > 85) Color.WHITE else Color.BLACK
        }
        export.setPixels(pixels, 0, w, 0, 0, w, h)

        return export
    }

    fun clearMask() {
        baseMaskBitmap?.recycle()
        baseMaskBitmap = null
        actionStack.clear()
        redoStack.clear()
        redrawHistory()
    }
    
    fun undo() {
        if (actionStack.isNotEmpty()) {
            redoStack.add(actionStack.removeAt(actionStack.lastIndex))
            redrawHistory()
        }
    }
    
    fun redo() {
        if (redoStack.isNotEmpty()) {
            actionStack.add(redoStack.removeAt(redoStack.lastIndex))
            redrawHistory()
        }
    }
    
    private fun redrawHistory() {
        displayMaskBitmap?.eraseColor(Color.TRANSPARENT)

        baseMaskBitmap?.let { base ->
            val paint = Paint().apply {
                colorFilter = PorterDuffColorFilter(maskColor, PorterDuff.Mode.SRC_IN)
            }
            displayMaskCanvas?.drawBitmap(base, 0f, 0f, paint)
        }

        val paint = Paint(basePaint).apply { color = maskColor }
        val eraser = Paint(basePaint).apply { xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR) }
        
        for (action in actionStack) {
            paint.strokeWidth = action.size
            eraser.strokeWidth = action.size
            
            when (action.mode) {
                ToolMode.BRUSH -> { paint.style = Paint.Style.STROKE; displayMaskCanvas?.drawPath(action.path, paint) }
                ToolMode.LASSO -> { paint.style = Paint.Style.FILL_AND_STROKE; displayMaskCanvas?.drawPath(action.path, paint) }
                ToolMode.ERASER -> { eraser.style = Paint.Style.STROKE; displayMaskCanvas?.drawPath(action.path, eraser) }
            }
        }
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val cursorX = touchX
        val cursorY = touchY - cursorOffset

        // 1. Draw Main Canvas
        displayMaskBitmap?.let {
            canvas.save()
            // View-level pinch zoom/pan, then the fit-center mapping.
            canvas.translate(viewPanX, viewPanY)
            canvas.scale(viewZoom, viewZoom)
            canvas.translate(offsetX, offsetY)
            canvas.scale(scaleFactor, scaleFactor)
            canvas.drawBitmap(it, 0f, 0f, null)
            
            if (isDrawing) {
                val livePaint = Paint(basePaint).apply {
                    color = maskColor
                    strokeWidth = brushSize
                    style = Paint.Style.STROKE
                    if (currentMode == ToolMode.ERASER) xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
                }
                canvas.drawPath(currentPath, livePaint)
            }
            canvas.restore()
        }
        
        // 2. Draw Cursor Hollow Circle (Main View) - hidden while pinching
        if (!isPinching && (isDrawing || touchX != -1f)) {
            val displayBrushRadius = (brushSize * scaleFactor * viewZoom) / 2f
            canvas.drawCircle(cursorX, cursorY, displayBrushRadius, cursorPaint)
        }

        // 3. Draw Smart Loupe (Magnifier) - hidden while pinching
        if (!isPinching && (isDrawing || touchX != -1f)) {
            val loupeRadius = 180f
            val margin = 60f
            
            // Smart placement: Keep it away from the hand
            val loupeCenterX = if (touchX < width / 2) width - loupeRadius - margin else loupeRadius + margin
            val loupeCenterY = loupeRadius + margin

            canvas.save()
            // Move drawing center to the Loupe position
            canvas.translate(loupeCenterX, loupeCenterY)
            
            // Clip everything to a perfect circle
            val loupePath = Path().apply { addCircle(0f, 0f, loupeRadius, Path.Direction.CW) }
            canvas.clipPath(loupePath)
            
            // Draw dark background in case zooming goes off the edge of the photo
            canvas.drawColor(Color.parseColor("#1A1A1A"))
            
            // Zoom in 2.5x and shift so the cursor acts as the dead-center
            val zoom = 2.5f
            canvas.scale(zoom, zoom)
            canvas.translate(-cursorX, -cursorY)
            
            // Redraw the photo & masks inside the loupe
            imageBitmap?.let {
                canvas.save()
                canvas.translate(viewPanX, viewPanY)
                canvas.scale(viewZoom, viewZoom)
                canvas.translate(offsetX, offsetY)
                canvas.scale(scaleFactor, scaleFactor)
                canvas.drawBitmap(it, 0f, 0f, null)
                displayMaskBitmap?.let { mask ->
                    canvas.drawBitmap(mask, 0f, 0f, null)
                }
                if (isDrawing) {
                    val livePaint = Paint(basePaint).apply {
                        color = maskColor
                        strokeWidth = brushSize
                        style = Paint.Style.STROKE
                        if (currentMode == ToolMode.ERASER) xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
                    }
                    canvas.drawPath(currentPath, livePaint)
                }
                canvas.restore()
            }
            
            // Draw precise targeting Crosshair
            val crosshairLen = 20f / zoom
            crosshairPaint.strokeWidth = 4f / zoom // Keep lines sharp
            canvas.drawLine(cursorX - crosshairLen, cursorY, cursorX + crosshairLen, cursorY, crosshairPaint)
            canvas.drawLine(cursorX, cursorY - crosshairLen, cursorX, cursorY + crosshairLen, crosshairPaint)
            
            // Draw Cursor Hollow Circle (Loupe View)
            val displayBrushRadius = (brushSize * scaleFactor * viewZoom) / 2f
            val loupeCursorPaint = Paint(cursorPaint).apply { strokeWidth = 5f / zoom }
            canvas.drawCircle(cursorX, cursorY, displayBrushRadius, loupeCursorPaint)

            canvas.restore()
            
            // Finally, draw the bold border ring around the Loupe
            canvas.drawCircle(loupeCenterX, loupeCenterY, loupeRadius, loupeBorderPaint)
        }
    }
    
    private fun mapToImage(x: Float, y: Float): Pair<Float, Float>? {
        // Undo the view-level zoom/pan first, then the fit-center mapping.
        val fitX = (x - viewPanX) / viewZoom
        val fitY = (y - viewPanY) / viewZoom
        if (!displayRect.contains(fitX, fitY)) return null
        return Pair((fitX - offsetX) / scaleFactor, (fitY - offsetY) / scaleFactor)
    }
    
    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                touchX = event.x
                touchY = event.y
                isPinching = false
                // Apply Offset!
                val drawY = event.y - cursorOffset
                val mapped = mapToImage(event.x, drawY)
                if (mapped != null) {
                    isDrawing = true
                    currentPath = Path()
                    currentPath.moveTo(mapped.first, mapped.second)
                    currentPath.lineTo(mapped.first, mapped.second)
                    redoStack.clear()
                }
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (event.pointerCount >= 2) {
                    // Two fingers: abandon any in-progress stroke and switch to
                    // zoom/pan until every extra finger is lifted.
                    if (isDrawing) {
                        isDrawing = false
                        currentPath = Path()
                    }
                    isPinching = true
                    touchX = -1f
                    pinchStartSpan = pinchSpan(event)
                    pinchStartZoom = viewZoom
                    pinchStartPanX = viewPanX
                    pinchStartPanY = viewPanY
                    val focal = pinchFocal(event)
                    pinchStartFocalX = focal.first
                    pinchStartFocalY = focal.second
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (isPinching && event.pointerCount >= 2) {
                    val span = pinchSpan(event)
                    val focal = pinchFocal(event)
                    val newZoom = (pinchStartZoom * span / pinchStartSpan).coerceIn(1f, 8f)
                    // Fit-space point that was under the starting focal stays
                    // under the current focal while zooming and panning.
                    val fitX = (pinchStartFocalX - pinchStartPanX) / pinchStartZoom
                    val fitY = (pinchStartFocalY - pinchStartPanY) / pinchStartZoom
                    viewZoom = newZoom
                    viewPanX = focal.first - newZoom * fitX
                    viewPanY = focal.second - newZoom * fitY
                    clampPan()
                    notifyTransform()
                } else if (!isPinching) {
                    touchX = event.x
                    touchY = event.y
                    val drawY = event.y - cursorOffset
                    val mapped = mapToImage(event.x, drawY)
                    if (isDrawing && mapped != null) {
                        currentPath.lineTo(mapped.first, mapped.second)
                    }
                }
            }
            MotionEvent.ACTION_POINTER_UP -> {
                if (isPinching && event.pointerCount - 1 <= 1) {
                    // Back to one finger: stay in "no drawing" state until the
                    // next fresh touch so the stroke doesn't jump.
                    isPinching = false
                    isDrawing = false
                    touchX = -1f
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (isPinching) {
                    isPinching = false
                } else if (isDrawing) {
                    if (currentMode == ToolMode.LASSO) currentPath.close()
                    // Save to history
                    actionStack.add(DrawAction(Path(currentPath), currentMode, brushSize))
                    isDrawing = false
                    redrawHistory()
                }
                touchX = -1f // Hide cursor
            }
        }
        invalidate()
        return true
    }
}
