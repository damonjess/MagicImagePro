package com.example.magicimagepro.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

class ObjectRemover {
    
    // Using Stable Diffusion v1.5 Inpainting model on Hugging Face
    private val apiEndpoint = "https://router.huggingface.co/hf-inference/models/stable-diffusion-v1-5/stable-diffusion-inpainting" 
    
    // Hugging Face API key loaded securely from local.properties via BuildConfig
    private val apiKey = com.example.magicimagepro.BuildConfig.HF_API_KEY

    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    suspend fun removeObject(image: Bitmap, maskBitmap: Bitmap): Bitmap = withContext(Dispatchers.IO) {
        
        val imageBytes = bitmapToByteArray(image)
        val maskBytes = bitmapToByteArray(maskBitmap)

        // Hugging Face uses slightly different names for the parameters
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "image.jpg", imageBytes.toRequestBody("image/jpeg".toMediaType()))
            .addFormDataPart("mask_image", "mask.jpg", maskBytes.toRequestBody("image/jpeg".toMediaType()))
            .addFormDataPart("inputs", "seamless background, highly detailed, photorealistic, exact match")
            .build()

        val request = Request.Builder()
            .url(apiEndpoint)
            // THIS IS THE FIX: Hugging Face requires the Authorization: Bearer format
            .addHeader("Authorization", "Bearer $apiKey")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                throw Exception("HF Cloud Error ${response.code}: $errorBody")
            }

            val responseBodyBytes = response.body?.bytes() 
                ?: throw Exception("Empty response from cloud server")

            BitmapFactory.decodeByteArray(responseBodyBytes, 0, responseBodyBytes.size)
                ?: throw Exception("Failed to decode cloud image response")
        }
    }

    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
        return stream.toByteArray()
    }
}
