package io.github.guzinski.farol

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import io.github.guzinski.farol.ui.FarolApp

class MainActivity : ComponentActivity() {
    private lateinit var controller: FarolController
    private var accepted by mutableStateOf(false)
    private var cameraGranted by mutableStateOf(false)
    private val preferences by lazy { getSharedPreferences("farol", MODE_PRIVATE) }
    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        cameraGranted = granted
        controller.permission(granted)
    }
    private val chooseVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(controller::openReplay)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowCompat.getInsetsController(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }

        accepted = preferences.getBoolean("disclaimer", false)
        controller = FarolController(this)
        updatePermission()
        setContent {
            FarolApp(
                controller = controller,
                accepted = accepted,
                cameraGranted = cameraGranted,
                accept = {
                    accepted = true
                    preferences.edit { putBoolean("disclaimer", true) }
                    updatePermission()
                    if (!cameraGranted) requestCamera.launch(Manifest.permission.CAMERA)
                },
                requestCamera = { requestCamera.launch(Manifest.permission.CAMERA) },
                chooseVideo = { chooseVideo.launch(arrayOf("video/*")) },
            )
        }
    }

    private fun updatePermission() {
        cameraGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        controller.permission(accepted && cameraGranted)
    }

    override fun onStart() {
        super.onStart()
        updatePermission()
        controller.setForeground(true)
    }

    override fun onStop() {
        controller.setForeground(false)
        super.onStop()
    }

    override fun onDestroy() {
        controller.close()
        super.onDestroy()
    }
}
