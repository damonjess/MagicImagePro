package com.example.magicimagepro

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.lifecycle.lifecycleScope
import com.example.magicimagepro.databinding.ActivityMainBinding
import com.example.magicimagepro.ml.ImageUpscaler
import com.example.magicimagepro.ml.NativeProcessor
import com.example.magicimagepro.ml.ObjectRemover
import com.example.magicimagepro.ml.ObjectSnapper
import com.example.magicimagepro.ui.ToolMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private var currentBitmap: Bitmap? = null
    private var objectRemover: ObjectRemover? = null
    @Suppress("unused")
    private var objectSnapper: ObjectSnapper? = null
    @Suppress("unused")
    private var imageUpscaler: ImageUpscaler? = null
    private var mInterstitialAd: Any? = null
    private val nativeProcessor = NativeProcessor()
    
    private val activeColor = Color.parseColor("#3DDC84")
    private val inactiveColor = Color.parseColor("#777777")
    
    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { 
            loadImage(it)
        }
    }
    
    private val cameraPicker = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        bitmap?.let {
            currentBitmap = it
            binding.imageView.setImageBitmap(it)
            binding.maskView.setImage(it)
            updateEmptyState(true)
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        applyWindowInsets()
        
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                objectRemover = ObjectRemover(this@MainActivity)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        
        setupUI()
        updateEmptyState(false)
    }
    
    private fun setupUI() {
        // New Camera Button
        binding.btnCamera.setOnClickListener { cameraPicker.launch(null) }
        
        // New Load Button
        binding.btnLoad.setOnClickListener { imagePicker.launch("image/*") }
        
        // Central Empty State Button
        binding.btnEmptyState.setOnClickListener { imagePicker.launch("image/*") }
        
        binding.btnSave.setOnClickListener {
            val bitmap = currentBitmap
            if (bitmap != null) {
                saveImageToGallery(bitmap)
            } else {
                Toast.makeText(this, "No image to save", Toast.LENGTH_SHORT).show()
            }
        }
        
        binding.btnUndo.setOnClickListener { binding.maskView.undo() }
        binding.btnRedo.setOnClickListener { binding.maskView.redo() }
        
        binding.sizeSlider.addOnChangeListener { _, value, _ -> binding.maskView.brushSize = value }
        binding.offsetSlider.addOnChangeListener { _, value, _ -> binding.maskView.cursorOffset = value }
        
        binding.btnBrush.setOnClickListener { setTool(ToolMode.BRUSH) }
        binding.btnLasso.setOnClickListener { setTool(ToolMode.LASSO) }
        binding.btnEraser.setOnClickListener { setTool(ToolMode.ERASER) }
        
        binding.btnProcess.setOnClickListener {
            val image = currentBitmap ?: run {
                Toast.makeText(this, "Please select or take a photo first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val rawMask = binding.maskView.getMaskBitmap() ?: return@setOnClickListener

            if (isMaskEmpty(rawMask)) {
                Toast.makeText(this, "Please draw a mask over the object to remove", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            binding.progressBar.visibility = View.VISIBLE

            val deferredResult = lifecycleScope.async(Dispatchers.Default) {
                try {
                    val mask = if (rawMask.width != image.width || rawMask.height != image.height) {
                        Bitmap.createScaledBitmap(rawMask, image.width, image.height, true)
                    } else {
                        rawMask
                    }
                    val remover = objectRemover
                    val result = if (remover != null) {
                        remover.removeObject(image, mask)
                    } else {
                        val fallback = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                        val status = nativeProcessor.processImage(image, mask, fallback)
                        if (status != 0) {
                            fallback.recycle()
                            error("Native inpainting failed with status $status")
                        }
                        fallback
                    }
                    if (mask != rawMask) {
                        mask.recycle()
                    }
                    Result.success(result)
                } catch (e: Exception) {
                    Result.failure(e)
                }
            }

            lifecycleScope.launch(Dispatchers.Main) {
                val result = deferredResult.await()

                result.onSuccess { bitmap ->
                    currentBitmap = bitmap
                    binding.imageView.setImageBitmap(bitmap)
                    binding.maskView.setImage(bitmap)
                }.onFailure { e ->
                    e.printStackTrace()
                    val errorMsg = e.localizedMessage ?: e::class.simpleName ?: "Unknown error"
                    Toast.makeText(this@MainActivity, "Inference Error: $errorMsg", Toast.LENGTH_LONG).show()
                }

                binding.progressBar.visibility = View.GONE
            }
        }
    }
    
    private fun isMaskEmpty(mask: Bitmap): Boolean {
        val w = mask.width
        val h = mask.height
        val step = maxOf(1, maxOf(w, h) / 200)
        val pixels = IntArray(w * h)
        mask.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in 0 until w * h step step) {
            val c = pixels[i]
            // Luminance only: the mask background is opaque black (alpha 255), so an alpha
            // check would never detect an empty mask. White = something was drawn.
            if (Color.red(c) > 50 || Color.green(c) > 50 || Color.blue(c) > 50) {
                return false
            }
        }
        return true
    }

    private fun saveImageToGallery(bitmap: Bitmap) {
        try {
            val filename = "MagicImagePro_${System.currentTimeMillis()}.png"
            val resolver = contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MagicImagePro")
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
            }

            val imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            if (imageUri != null) {
                resolver.openOutputStream(imageUri)?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                    resolver.update(imageUri, contentValues, null, null)
                }
                Toast.makeText(this, "Saved to Pictures/MagicImagePro", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error saving image: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setTool(mode: ToolMode) {
        binding.maskView.currentMode = mode
        binding.btnBrush.setTextColor(if (mode == ToolMode.BRUSH) activeColor else inactiveColor)
        binding.btnLasso.setTextColor(if (mode == ToolMode.LASSO) activeColor else inactiveColor)
        binding.btnEraser.setTextColor(if (mode == ToolMode.ERASER) activeColor else inactiveColor)
    }
    
    private fun loadImage(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            val inputStream = contentResolver.openInputStream(uri)
            val rawBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            // Downscale massive 4000px camera photos to max 1600px
            val maxDimension = 1600
            val bitmap = if (rawBitmap != null && (rawBitmap.width > maxDimension || rawBitmap.height > maxDimension)) {
                val scale = maxDimension.toFloat() / maxOf(rawBitmap.width, rawBitmap.height)
                val w = (rawBitmap.width * scale).toInt()
                val h = (rawBitmap.height * scale).toInt()
                Bitmap.createScaledBitmap(rawBitmap, w, h, true).also { rawBitmap.recycle() }
            } else {
                rawBitmap
            }

            withContext(Dispatchers.Main) {
                bitmap?.let {
                    currentBitmap = it
                    binding.imageView.setImageBitmap(it)
                    binding.maskView.setImage(it)
                    updateEmptyState(true)
                }
            }
        }
    }

    private fun updateEmptyState(isImageLoaded: Boolean) {
        binding.btnEmptyState.visibility = if (isImageLoaded) View.GONE else View.VISIBLE
        binding.imageView.visibility = if (isImageLoaded) View.VISIBLE else View.GONE
        binding.maskView.visibility = if (isImageLoaded) View.VISIBLE else View.GONE
    }

    private fun applyWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(binding.topBar) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.updatePadding(top = systemBars.top)
            insets
        }
        ViewCompat.setOnApplyWindowInsetsListener(binding.bottomSheet) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.updatePadding(bottom = systemBars.bottom)
            insets
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Free the native C++ memory held by TensorFlow Lite
        objectRemover?.close()
        objectSnapper?.close()
        imageUpscaler?.close()
    }
}