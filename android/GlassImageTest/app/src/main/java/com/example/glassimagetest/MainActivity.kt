package com.example.glassimagetest

import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.example.glassimagetest.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Load test image from assets
        val baboonBitmap = this.assets.open(baboonPath).use { BitmapFactory.decodeStream(it) }

        // Example of a call to a native method
        // binding.sampleText.text = stringFromJNI()

        // Display output image on screen
        binding.imageView.setImageBitmap(baboonBitmap)
    }

    /**
     * A native method that is implemented by the 'glassimagetest' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'glassimagetest' library on application startup.
        init {
            System.loadLibrary("glassimagetest")
        }

        private const val baboonPath = "baboon.png"
    }
}