package com.example.magicimagepro.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

abstract class TFLiteModel(context: Context, modelNameOrPath: String) {
    
    protected var interpreter: Interpreter

    init {
        val modelBuffer = loadModelFile(context, modelNameOrPath)
        
        // CRITICAL FIX: Removed GpuDelegate entirely.
        // LaMa uses Fourier Convolutions (FFTs). Mobile GPUs corrupt frequency-domain math, 
        // resulting in the green checkerboard static. We MUST use the CPU.
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        
        interpreter = Interpreter(modelBuffer, options)
    }
    
    private fun loadModelFile(context: Context, path: String): MappedByteBuffer {
        if (path.startsWith("/")) {
            val file = File(path)
            val inputStream = FileInputStream(file)
            val fileChannel = inputStream.channel
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        } else {
            val fileDescriptor = context.assets.openFd(path)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }
    
    fun close() {
        interpreter.close()
    }
}
