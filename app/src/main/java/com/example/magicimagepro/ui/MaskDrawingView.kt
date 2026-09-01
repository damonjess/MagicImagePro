package com.example.magicimagepro.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import androidx.core.view.doOnLayout

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

    private var imageBitmap: Bitmap? = null
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

    fun setImage(bitmap: Bitmap) {
        imageBitmap = bitmap
        
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
        }
    }

    fun setMask(mask: Bitmap) {
        val bmp = imageBitmap ?: return
        val scaledMask = if (mask.width != bmp.width || mask.height != bmp.height) {
            Bitmap.createScaledBitmap(mask, bmp.width, bmp.height, true)
        } else {
            mask
        }

        actionStack.clear()
        redoStack.clear()

        exportMaskBitmap = scaledMask.copy(Bitmap.Config.ARGB_8888, true)
        exportMaskCanvas = Canvas(exportMaskBitmap!!)
        
        displayMaskBitmap?.eraseColor(Color.TRANSPARENT)
        val paint = Paint().apply {
            colorFilter = PorterDuffColorFilter(maskColor, PorterDuff.Mode.SRC_IN)
        }
        displayMaskCanvas?.drawBitmap(exportMaskBitmap!!, 0f, 0f, paint)
        invalidate()
    }
    
    fun getMaskBitmap(): Bitmap? {
        exportMaskCanvas?.drawColor(Color.BLACK, PorterDuff.Mode.CLEAR)
        exportMaskCanvas?.drawColor(Color.BLACK) // Black background
        
        val exportPaint = Paint(basePaint).apply { color = Color.WHITE }
        val eraserPaint = Paint(basePaint).apply { xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR) }
        
        for (action in actionStack) {
            exportPaint.strokeWidth = action.size
            eraserPaint.strokeWidth = action.size
            
            when (action.mode) {
                ToolMode.BRUSH -> { exportPaint.style = Paint.Style.STROKE; exportMaskCanvas?.drawPath(action.path, exportPaint) }
                ToolMode.LASSO -> { exportPaint.style = Paint.Style.FILL_AND_STROKE; exportMaskCanvas?.drawPath(action.path, exportPaint) }
                ToolMode.ERASER -> { eraserPaint.style = Paint.Style.STROKE; exportMaskCanvas?.drawPath(action.path, eraserPaint) }
            }
        }
        return exportMaskBitmap
    }

    fun clearMask() {
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
        
        // 2. Draw Cursor Hollow Circle (Main View)
        if (isDrawing || touchX != -1f) {
            val displayBrushRadius = (brushSize * scaleFactor) / 2f
            canvas.drawCircle(cursorX, cursorY, displayBrushRadius, cursorPaint)
        }

        // 3. Draw Smart Loupe (Magnifier)
        if (isDrawing || touchX != -1f) {
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
            val displayBrushRadius = (brushSize * scaleFactor) / 2f
            val loupeCursorPaint = Paint(cursorPaint).apply { strokeWidth = 5f / zoom }
            canvas.drawCircle(cursorX, cursorY, displayBrushRadius, loupeCursorPaint)

            canvas.restore()
            
            // Finally, draw the bold border ring around the Loupe
            canvas.drawCircle(loupeCenterX, loupeCenterY, loupeRadius, loupeBorderPaint)
        }
    }
    
    private fun mapToImage(x: Float, y: Float): Pair<Float, Float>? {
        if (!displayRect.contains(x, y)) return null
        return Pair((x - offsetX) / scaleFactor, (y - offsetY) / scaleFactor)
    }
    
    override fun onTouchEvent(event: MotionEvent): Boolean {
        touchX = event.x
        touchY = event.y
        
        // Apply Offset!
        val drawY = event.y - cursorOffset
        val mapped = mapToImage(event.x, drawY)
        
        if (editMode == EditMode.SMART_SELECT) {
            if (event.action == MotionEvent.ACTION_UP) {
                val upMapped = mapToImage(event.x, event.y) // Tap is where finger is, not offset
                if (upMapped != null) {
                    onTapListener?.onTap(upMapped.first, upMapped.second)
                }
            }
            invalidate()
            return true
        }

        if (mapped == null && event.action != MotionEvent.ACTION_UP) {
            invalidate(); return true 
        }
        
        val mx = mapped?.first ?: 0f
        val my = mapped?.second ?: 0f
        
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDrawing = true
                currentPath = Path()
                currentPath.moveTo(mx, my)
                redoStack.clear()
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDrawing) currentPath.lineTo(mx, my)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (isDrawing) {
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
