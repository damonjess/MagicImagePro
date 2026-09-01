package com.example.magicimagepro

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.doOnLayout
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
    private var isMaskMode = false
    
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
        
        // Load ML models in background
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                objectRemover = ObjectRemover(this@MainActivity)
                imageUpscaler = ImageUpscaler(this@MainActivity)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Model load failed: ${e.message}", Toast.LENGTH_LONG).show()
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
            val bitmap = currentBitmap ?: run {
                Toast.makeText(this, "Load an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            isMaskMode = !isMaskMode
            
            if (isMaskMode) {
                // Enter mask mode
                binding.maskView.visibility = View.VISIBLE
                binding.modeIndicator.visibility = View.VISIBLE
                binding.btnMask.text = "Done"
                binding.btnMask.setBackgroundColor(getColor(R.color.purple_500))
                
                // Important: wait for layout before setting image so display math works
                binding.maskView.doOnLayout {
                    binding.maskView.setImage(bitmap)
                }
                Toast.makeText(this, "Paint over objects to remove", Toast.LENGTH_SHORT).show()
            } else {
                // Exit mask mode
                binding.maskView.visibility = View.GONE
                binding.modeIndicator.visibility = View.GONE
                binding.btnMask.text = "Mask"
                binding.btnMask.setBackgroundColor(getColor(R.color.purple_200))
            }
        }
        
        binding.btnRemove.setOnClickListener {
            val image = currentBitmap ?: run {
                Toast.makeText(this, "Load an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val mask = binding.maskView.getMaskBitmap() ?: run {
                Toast.makeText(this, "Draw a mask first (tap Mask)", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            // Exit mask mode
            isMaskMode = false
            binding.maskView.visibility = View.GONE
            binding.modeIndicator.visibility = View.GONE
            binding.btnMask.text = "Mask"
            binding.btnMask.setBackgroundColor(getColor(R.color.purple_200))
            
            showLoading(true)
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val result = objectRemover?.removeObject(image, mask)
                    withContext(Dispatchers.Main) {
                        result?.let {
                            currentBitmap = it
                            binding.imageView.setImageBitmap(it)
                            binding.maskView.clearMask()
                        }
                        showLoading(false)
                        Toast.makeText(this@MainActivity, "Object removed!", Toast.LENGTH_SHORT).show()
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
            val image = currentBitmap ?: run {
                Toast.makeText(this, "Load an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
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
                ?: Toast.makeText(this, "No image to save", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun loadImage(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
                
                withContext(Dispatchers.Main) {
                    currentBitmap = bitmap
                    binding.imageView.setImageBitmap(bitmap)
                    // Reset mask mode
                    isMaskMode = false
                    binding.maskView.visibility = View.GONE
                    binding.modeIndicator.visibility = View.GONE
                    binding.btnMask.text = "Mask"
                    binding.btnMask.setBackgroundColor(getColor(R.color.purple_200))
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    private fun saveImage(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
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
                    contentResolver.openOutputStream(it)?.use { stream ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                    }
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Saved: $filename", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Save failed: ${e.message}", Toast.LENGTH_SHORT).show()
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
