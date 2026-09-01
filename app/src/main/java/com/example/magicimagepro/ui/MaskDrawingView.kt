package com.example.magicimagepro.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class MaskDrawingView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    
    private var bitmap: Bitmap? = null
    private var canvas: Canvas? = null
    private val paint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 60f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
    }
    
    private val path = Path()
    private var scaleFactor = 1f
    private var translateX = 0f
    private var translateY = 0f
    
    fun setImage(bitmap: Bitmap) {
        this.bitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        canvas = Canvas(this.bitmap!!)
        canvas?.drawColor(Color.BLACK) // Black = keep, White = remove
        
        // Scale to fit view while maintaining aspect ratio
        val scaleX = width.toFloat() / bitmap.width
        val scaleY = height.toFloat() / bitmap.height
        scaleFactor = minOf(scaleX, scaleY)
        
        val scaledWidth = bitmap.width * scaleFactor
        val scaledHeight = bitmap.height * scaleFactor
        translateX = (width - scaledWidth) / 2f
        translateY = (height - scaledHeight) / 2f
        
        invalidate()
    }
    
    fun getMaskBitmap(): Bitmap? = bitmap
    
    fun clearMask() {
        canvas?.drawColor(Color.BLACK)
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        bitmap?.let {
            canvas.save()
            canvas.translate(translateX, translateY)
            canvas.scale(scaleFactor, scaleFactor)
            canvas.drawBitmap(it, 0f, 0f, null)
            canvas.restore()
        }
    }
    
    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = (event.x - translateX) / scaleFactor
        val y = (event.y - translateY) / scaleFactor
        
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                path.moveTo(x, y)
                canvas?.drawPoint(x, y, paint)
            }
            MotionEvent.ACTION_MOVE -> {
                path.lineTo(x, y)
                canvas?.drawPath(path, paint)
                path.reset()
                path.moveTo(x, y)
            }
            MotionEvent.ACTION_UP -> {
                path.reset()
            }
        }
        invalidate()
        return true
    }
}
