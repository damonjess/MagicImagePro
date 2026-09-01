package com.example.magicimagepro.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

abstract class TFLiteModel(context: Context, modelPath: String) {
    
    protected val interpreter: Interpreter
    
    init {
        val options = Interpreter.Options().apply {
            // Honor Magic 8 Pro has Adreno GPU — use it
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                addDelegate(GpuDelegate())
            }
            setNumThreads(4)
        }
        interpreter = Interpreter(loadModelFile(context, modelPath), options)
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun close() = interpreter.close()
}
