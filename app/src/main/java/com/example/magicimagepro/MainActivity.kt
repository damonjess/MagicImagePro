package com.example.magicimagepro

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.magicimagepro.databinding.ActivityMainBinding
import com.example.magicimagepro.ml.ImageUpscaler
import com.example.magicimagepro.ml.ObjectRemover
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStream

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private var currentBitmap: Bitmap? = null
    private var objectRemover: ObjectRemover? = null
    private var imageUpscaler: ImageUpscaler? = null
    
    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadImage(it) }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setSupportActionBar(binding.toolbar)
        
        // Initialize ML models (heavy operation — do in background)
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                objectRemover = ObjectRemover(this@MainActivity)
                imageUpscaler = ImageUpscaler(this@MainActivity)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Model Init Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
        
        setupButtons()
    }
    
    private fun setupButtons() {
        binding.btnLoad.setOnClickListener {
            imagePicker.launch("image/*")
        }
        
        binding.btnMask.setOnClickListener {
            currentBitmap?.let { bitmap ->
                binding.maskView.visibility = View.VISIBLE
                binding.maskView.setImage(bitmap)
                Toast.makeText(this, "Paint over objects to remove", Toast.LENGTH_SHORT).show()
            }
        }
        
        binding.btnRemove.setOnClickListener {
            val image = currentBitmap ?: return@setOnClickListener
            val mask = binding.maskView.getMaskBitmap() ?: return@setOnClickListener
            
            showLoading(true)
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val result = objectRemover?.removeObject(image, mask)
                    withContext(Dispatchers.Main) {
                        result?.let {
                            currentBitmap = it
                            binding.imageView.setImageBitmap(it)
                            binding.maskView.visibility = View.GONE
                            binding.maskView.clearMask()
                        }
                        showLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                        showLoading(false)
                    }
                }
            }
        }
        
        binding.btnUpscale.setOnClickListener {
            val image = currentBitmap ?: return@setOnClickListener
            
            showLoading(true)
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val result = imageUpscaler?.upscale(image)
                    withContext(Dispatchers.Main) {
                        result?.let {
                            currentBitmap = it
                            binding.imageView.setImageBitmap(it)
                            Toast.makeText(this@MainActivity, "Upscaled to ${it.width}x${it.height}", Toast.LENGTH_SHORT).show()
                        }
                        showLoading(false)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Upscale Error: ${e.message}", Toast.LENGTH_LONG).show()
                        showLoading(false)
                    }
                }
            }
        }
        
        binding.btnSave.setOnClickListener {
            currentBitmap?.let { saveImage(it) }
        }
    }
    
    private fun loadImage(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            
            withContext(Dispatchers.Main) {
                currentBitmap = bitmap
                binding.imageView.setImageBitmap(bitmap)
                binding.maskView.setImage(bitmap!!)
            }
        }
    }
    
    private fun saveImage(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            val filename = "MagicImage_${System.currentTimeMillis()}.jpg"
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MagicImagePro")
            }
            
            val uri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            
            uri?.let {
                val outputStream: OutputStream? = contentResolver.openOutputStream(it)
                outputStream?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                }
                
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Saved: $filename", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    private fun showLoading(show: Boolean) {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
    }
    
    override fun onDestroy() {
        super.onDestroy()
        objectRemover?.close()
        imageUpscaler?.close()
    }
}
