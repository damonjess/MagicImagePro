package com.example.magicimagepro.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import androidx.core.view.doOnLayout

class MaskDrawingView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    
    private var imageBitmap: Bitmap? = null
    private var maskBitmap: Bitmap? = null
    private var maskCanvas: Canvas? = null
    
    private val maskPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 80f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
    }
    
    private val overlayPaint = Paint().apply {
        color = Color.RED
        alpha = 140
    }
    
    private var isDrawing = false
    private var lastMaskX = 0f
    private var lastMaskY = 0f
    
    // Display mapping
    private var scaleFactor = 1f
    private var offsetX = 0f
    private var offsetY = 0f
    private val displayRect = RectF()
    
    fun setImage(bitmap: Bitmap) {
        imageBitmap = bitmap
        
        // Create mask at original image resolution
        maskBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        maskCanvas = Canvas(maskBitmap!!)
        maskCanvas?.drawColor(Color.BLACK)
        
        // Wait for layout then calculate display mapping
        if (width > 0 && height > 0) {
            calculateDisplayRect()
            invalidate()
        } else {
            doOnLayout { calculateDisplayRect() }
        }
    }
    
    private fun calculateDisplayRect() {
        imageBitmap?.let { bmp ->
            if (width == 0 || height == 0) return
            
            val viewW = width.toFloat()
            val viewH = height.toFloat()
            val imgW = bmp.width.toFloat()
            val imgH = bmp.height.toFloat()
            
            // fitCenter math
            scaleFactor = minOf(viewW / imgW, viewH / imgH)
            val scaledW = imgW * scaleFactor
            val scaledH = imgH * scaleFactor
            offsetX = (viewW - scaledW) / 2f
            offsetY = (viewH - scaledH) / 2f
            
            displayRect.set(offsetX, offsetY, offsetX + scaledW, offsetY + scaledH)
        }
    }
    
    fun getMaskBitmap(): Bitmap? = maskBitmap
    
    fun clearMask() {
        maskCanvas?.drawColor(Color.BLACK)
        invalidate()
    }
    
    fun setBrushSize(size: Float) {
        maskPaint.strokeWidth = size
        invalidate()
    }
    
    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        calculateDisplayRect()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Draw the mask bitmap as a red overlay, scaled to match the on-screen image
        maskBitmap?.let { mask ->
            canvas.save()
            canvas.translate(offsetX, offsetY)
            canvas.scale(scaleFactor, scaleFactor)
            canvas.drawBitmap(mask, 0f, 0f, overlayPaint)
            canvas.restore()
        }
    }
    
    private fun mapTouchToMask(x: Float, y: Float): Pair<Float, Float>? {
        if (!displayRect.contains(x, y)) return null
        val maskX = (x - offsetX) / scaleFactor
        val maskY = (y - offsetY) / scaleFactor
        return Pair(maskX, maskY)
    }
    
    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y
        
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                val mapped = mapTouchToMask(x, y) ?: return false
                lastMaskX = mapped.first
                lastMaskY = mapped.second
                isDrawing = true
                parent.requestDisallowInterceptTouchEvent(true)
            }
            
            MotionEvent.ACTION_MOVE -> {
                if (!isDrawing) return false
                val mapped = mapTouchToMask(x, y) ?: run {
                    isDrawing = false
                    invalidate()
                    return true
                }
                
                // Draw line on mask bitmap (at original resolution)
                maskCanvas?.drawLine(lastMaskX, lastMaskY, mapped.first, mapped.second, maskPaint)
                
                lastMaskX = mapped.first
                lastMaskY = mapped.second
                invalidate()
            }
            
            MotionEvent.ACTION_UP -> {
                isDrawing = false
                parent.requestDisallowInterceptTouchEvent(false)
                invalidate()
            }
        }
        return true
    }
}
