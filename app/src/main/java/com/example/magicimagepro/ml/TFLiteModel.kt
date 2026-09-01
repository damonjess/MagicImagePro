package com.example.magicimagepro.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

abstract class TFLiteModel(context: Context, modelNameOrPath: String) {
    
    protected var interpreter: Interpreter

    init {
        val modelBuffer = loadModelFile(context, modelNameOrPath)
        var tempInterpreter: Interpreter? = null

        // Try GPU Delegate first
        try {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val gpuOptions = Interpreter.Options().apply {
                    addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
                    setNumThreads(4)
                }
                tempInterpreter = Interpreter(modelBuffer, gpuOptions)
            }
        } catch (e: Exception) {
            Log.w("TFLiteModel", "GPU delegate failed, falling back to CPU", e)
        }

        // Fallback to CPU with multi-threading
        if (tempInterpreter == null) {
            tempInterpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
        }
        
        interpreter = tempInterpreter
    }
    
    private fun loadModelFile(context: Context, path: String): MappedByteBuffer {
        // If the string starts with a slash, load it from the phone's physical storage
        if (path.startsWith("/")) {
            val file = File(path)
            if (!file.exists()) {
                throw IllegalArgumentException("Model not found on phone storage at: $path")
            }
            val inputStream = FileInputStream(file)
            val fileChannel = inputStream.channel
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        } 
        // Otherwise, load it from the bundled APK assets folder
        else {
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
