package kr.co.car.campingfriend

import android.Manifest
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kr.co.car.campingfriend.databinding.ActivityMainBinding
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var sharedPreferences: SharedPreferences

    private val textRecognizer by lazy {
        TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // SharedPreferences 초기화
        sharedPreferences = getSharedPreferences("CampingAppPrefs", MODE_PRIVATE)

        // 카메라 권한 확인
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // 카메라 실행을 위한 Executor 생성
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 설정 버튼 클릭 리스너 추가
        val settingsButton: TextView = findViewById(R.id.settingsButton)
        settingsButton.setOnClickListener {
            showCampingIdDialog()
        }

        // 기존 저장된 캠핑장 ID 불러오기
        val savedCampingId = getSavedCampingId()
        viewBinding.detectedPlateText.text = "캠핑장 ID: $savedCampingId"
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                // 프리뷰 설정
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

                // 후면 카메라 선택
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                // 이미지 분석 설정
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                // 이미지 분석기 설정
                val licensePlateProcessor = ImprovedLicensePlateDetectionProcessor(
                    textRecognizer,
                    { plateNumber ->
                        runOnUiThread {
                            viewBinding.detectedPlateText.text = "인식된 번호판: $plateNumber"
                        }
                    },
                    { status ->
                        runOnUiThread {
                            viewBinding.serverStatusText.text = "서버 상태: $status"
                        }
                    }
                )

                imageAnalysis.setAnalyzer(cameraExecutor, licensePlateProcessor)

                // 이전 바인딩 해제 후 새 바인딩 적용
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )

            } catch (e: Exception) {
                Log.e(TAG, "카메라 프로세스 바인딩 실패", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showCampingIdDialog() {
        val editText = EditText(this)
        editText.setText(getSavedCampingId())

        AlertDialog.Builder(this)
            .setTitle("캠핑장 ID 입력")
            .setView(editText)
            .setPositiveButton("저장") { _, _ ->
                val campingId = editText.text.toString()
                saveCampingId(campingId)
                viewBinding.detectedPlateText.text = "캠핑장 ID: $campingId"
            }
            .setNegativeButton("취소", null)
            .show()
    }

    private fun saveCampingId(id: String) {
        sharedPreferences.edit().putString("camping_id", id).apply()
    }

    private fun getSavedCampingId(): String {
        return sharedPreferences.getString("camping_id", "") ?: "없음"
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.camera_permission_required),
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
