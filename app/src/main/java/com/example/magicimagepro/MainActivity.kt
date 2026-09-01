package com.example.magicimagepro

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
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
import com.example.magicimagepro.ml.ObjectRemover
import com.example.magicimagepro.ui.ToolMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private var currentBitmap: Bitmap? = null
    private var objectRemover: ObjectRemover? = null
    
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
        
        lifecycleScope.launch(Dispatchers.Default) {
            try { objectRemover = ObjectRemover(this@MainActivity) } 
            catch (e: Exception) {}
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
        
        binding.btnSave.setOnClickListener { Toast.makeText(this, "Add save logic", Toast.LENGTH_SHORT).show() }
        
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
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    val remover = objectRemover ?: throw Exception("Model loading")
                    val result = remover.removeObject(image, mask)
                    withContext(Dispatchers.Main) {
                        currentBitmap = result
                        binding.imageView.setImageBitmap(result)
                        binding.maskView.setImage(result) 
                        binding.progressBar.visibility = View.GONE
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) { 
                        Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                        binding.progressBar.visibility = View.GONE 
                    }
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
            val stream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(stream)
            stream?.close()
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