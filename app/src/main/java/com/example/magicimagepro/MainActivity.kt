package com.example.magicimagepro

import android.content.ContentValues
import android.content.Context
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
import com.example.magicimagepro.ml.NativeProcessor
import com.example.magicimagepro.ml.ObjectRemover
import com.example.magicimagepro.ui.ToolMode
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.FullScreenContentCallback
import com.google.android.gms.ads.LoadAdError
import com.google.android.gms.ads.MobileAds
import com.google.android.gms.ads.interstitial.InterstitialAd
import com.google.android.gms.ads.interstitial.InterstitialAdLoadCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private var currentBitmap: Bitmap? = null
    private var objectRemover: ObjectRemover? = null
    private var mInterstitialAd: InterstitialAd? = null
    
    private val activeColor = Color.parseColor("#3DDC84")
    private val inactiveColor = Color.parseColor("#777777")
    
    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { 
            loadImage(it)
            updateEmptyState(true)
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
        
        MobileAds.initialize(this) {}
        loadInterstitialAd()
        
        lifecycleScope.launch(Dispatchers.Default) {
            try { objectRemover = ObjectRemover() } 
            catch (e: Exception) {}
        }
        
        setupUI()
        updateEmptyState(false)
    }
    
    private fun loadInterstitialAd() {
        val adRequest = AdRequest.Builder().build()
        InterstitialAd.load(
            this,
            "ca-app-pub-3940256099942544/1033173712",
            adRequest,
            object : InterstitialAdLoadCallback() {
                override fun onAdLoaded(interstitialAd: InterstitialAd) {
                    mInterstitialAd = interstitialAd
                }
                override fun onAdFailedToLoad(adError: LoadAdError) {
                    mInterstitialAd = null
                }
            }
        )
    }
    
    private fun setupUI() {
        // New Camera Button
        binding.btnCamera.setOnClickListener { cameraPicker.launch(null) }
        
        // New Load Button
        binding.btnLoad.setOnClickListener { imagePicker.launch("image/*") }
        
        // Central Empty State Button
        binding.btnEmptyState.setOnClickListener { imagePicker.launch("image/*") }
        
        binding.btnSave.setOnClickListener {
            val imageToSave = currentBitmap
            if (imageToSave == null) {
                Toast.makeText(this, "No image to save!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            lifecycleScope.launch {
                val success = saveBitmapToGallery(this@MainActivity, imageToSave)
                if (success) {
                    Toast.makeText(this@MainActivity, "Saved to Gallery!", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, "Failed to save image.", Toast.LENGTH_SHORT).show()
                }
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
            val image = currentBitmap ?: return@setOnClickListener
            val mask = binding.maskView.getMaskBitmap() ?: return@setOnClickListener
            
            binding.progressBar.visibility = View.VISIBLE
            
            // 2. Safely run the API request in the background
            // By wrapping the response in Result<Bitmap>, we prevent network errors from crashing the app
            val deferredResult = lifecycleScope.async(Dispatchers.IO) {
                try {
                    val remover = ObjectRemover()
                    Result.success(remover.removeObject(image, mask))
                } catch (e: Exception) {
                    Result.failure(e)
                }
            }
            
            // 3. Show the ad to keep the user occupied
            if (mInterstitialAd != null) {
                mInterstitialAd?.fullScreenContentCallback = object: FullScreenContentCallback() {
                    override fun onAdDismissedFullScreenContent() {
                        mInterstitialAd = null
                        loadInterstitialAd() // Preload the next one invisibly
                        
                        // 4. Wait for the API to finish and safely unwrap the Result
                        lifecycleScope.launch(Dispatchers.Main) {
                            val result = deferredResult.await()
                            
                            result.onSuccess { bitmap ->
                                currentBitmap = bitmap
                                binding.imageView.setImageBitmap(bitmap)
                                binding.maskView.setImage(bitmap)
                                binding.maskView.clearMask()
                            }.onFailure { e ->
                                // This will print the exact reason the API failed to the screen!
                                Toast.makeText(this@MainActivity, "Cloud Error: ${e.message}", Toast.LENGTH_LONG).show()
                            }
                            
                            binding.progressBar.visibility = View.GONE
                        }
                    }
                }
                mInterstitialAd?.show(this@MainActivity)
            } else {
                // Fallback: If the ad didn't load in time, just wait for the API
                lifecycleScope.launch(Dispatchers.Main) {
                    val result = deferredResult.await()
                    
                    result.onSuccess { bitmap ->
                        currentBitmap = bitmap
                        binding.imageView.setImageBitmap(bitmap)
                        binding.maskView.setImage(bitmap)
                        binding.maskView.clearMask()
                    }.onFailure { e ->
                        Toast.makeText(this@MainActivity, "Cloud Error: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                    
                    binding.progressBar.visibility = View.GONE
                }
            }
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
                }
            }
        }
    }

    private fun updateEmptyState(isImageLoaded: Boolean) {
        binding.btnEmptyState.visibility = if (isImageLoaded) View.GONE else View.VISIBLE
        // Optionally show/hide top bar load buttons to reduce clutter when empty
        // binding.topBar.visibility = if (isImageLoaded) View.VISIBLE else View.INVISIBLE
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
}

suspend fun saveBitmapToGallery(context: Context, bitmap: Bitmap): Boolean = withContext(Dispatchers.IO) {
    val filename = "MagicImagePro_${System.currentTimeMillis()}.jpg"
    var outputStream: OutputStream? = null
    var imageUri: Uri? = null
    
    try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10+ (API 29+): Scoped Storage, no permissions required
            val contentValues = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/MagicImagePro")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            
            val resolver = context.contentResolver
            imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            
            imageUri?.let { uri ->
                outputStream = resolver.openOutputStream(uri)
                outputStream?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                }
                
                contentValues.clear()
                contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                resolver.update(uri, contentValues, null, null)
            }
        } else {
            // Android 9 and below: Requires WRITE_EXTERNAL_STORAGE permission
            val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val appDir = File(imagesDir, "MagicImagePro")
            if (!appDir.exists()) appDir.mkdirs()
            
            val imageFile = File(appDir, filename)
            outputStream = FileOutputStream(imageFile)
            outputStream?.use { stream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
            }
        }
        return@withContext true
    } catch (e: Exception) {
        e.printStackTrace()
        // Clean up the empty file if something crashed during the write process
        imageUri?.let { context.contentResolver.delete(it, null, null) }
        return@withContext false
    }
}
