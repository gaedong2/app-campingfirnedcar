package kr.co.car.campingfriend
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.media.Image
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognizer
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.regex.Pattern

class ImprovedLicensePlateDetectionProcessor(
    private val textRecognizer: TextRecognizer,
    private val plateNumberListener: (String) -> Unit,
    private val serverStatusListener: (String) -> Unit
) : ImageAnalysis.Analyzer {

    private val client = OkHttpClient()
    private val isProcessing = AtomicBoolean(false)
    private var lastDetectedPlate = ""
    private var lastDetectionTime = 0L
    private var lastSentTime = 0L

    // 최근 인식 결과 저장을 위한 맵 (번호판 -> 카운트)
    private val recentDetections = mutableMapOf<String, Int>()
    private val MAX_RECENT_DETECTIONS = 10
    private val CONFIDENCE_THRESHOLD = 0.7f

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        if (isProcessing.get()) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        isProcessing.set(true)

        // 이미지 전처리 (ImageProcessor 클래스 구현 필요)
        val processedImage = ImageProcessor.process(mediaImage, imageProxy.imageInfo.rotationDegrees)
        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        textRecognizer.process(image)
            .addOnSuccessListener { text ->
                processTextRecognitionResult(text, imageProxy)
                isProcessing.set(false)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "텍스트 인식 실패", e)
                isProcessing.set(false)
                imageProxy.close()
            }
    }

    private fun processTextRecognitionResult(text: Text, imageProxy: ImageProxy) {
        val possiblePlates = findPossibleLicensePlates(text)

        if (possiblePlates.isNotEmpty()) {
            // 신뢰도가 가장 높은 번호판 선택
            val (bestPlate, confidence) = possiblePlates.maxByOrNull { it.second } ?: Pair("", 0f)

            if (bestPlate.isNotEmpty() && confidence >= CONFIDENCE_THRESHOLD) {
                Log.d(TAG, "번호판 인식됨: $bestPlate (신뢰도: $confidence)")

                val currentTime = System.currentTimeMillis()

                // 번호판 중복 감지 방지 로직
                if (bestPlate == lastDetectedPlate && (currentTime - lastDetectionTime) < DETECTION_COOLDOWN_MS) {
                    Log.d(TAG, "쿨다운 시간 내 감지됨, 무시: $bestPlate")
                    imageProxy.close()
                    return
                }

                // 다중 프레임 검증 로직
                updateRecentDetections(bestPlate)
                val detectionCount = recentDetections[bestPlate] ?: 0

                if (detectionCount >= 3 || confidence > 0.9f) {
                    lastDetectedPlate = bestPlate
                    lastDetectionTime = currentTime
                    plateNumberListener(bestPlate)

                    // 서버 전송 딜레이 확인
                    if (currentTime - lastSentTime < SERVER_SEND_COOLDOWN_MS) {
                        Log.d(TAG, "서버 전송 쿨다운 시간 내, 무시: $bestPlate")
                        imageProxy.close()
                        return
                    }
                    lastSentTime = currentTime

                    when (IMAGE_SEND_MODE) {
                        "NONE" -> sendTextOnly(bestPlate)
                        "FULL_FRAME" -> sendFullFrame(bestPlate, imageProxy)
                        "CROPPED_PLATE" -> sendCroppedPlateImage(bestPlate, text, imageProxy)
                    }
                } else {
                    Log.d(TAG, "신뢰도 부족, 무시: $bestPlate (카운트: $detectionCount, 신뢰도: $confidence)")
                }
            }
        }
        imageProxy.close()
    }

    private fun updateRecentDetections(plate: String) {
        // 기존 카운트 가져오기
        val count = recentDetections[plate] ?: 0
        recentDetections[plate] = count + 1

        // 맵 크기 제한
        if (recentDetections.size > MAX_RECENT_DETECTIONS) {
            // 카운트가 가장 작은 항목 제거
            val minEntry = recentDetections.minByOrNull { it.value }
            minEntry?.let { recentDetections.remove(it.key) }
        }
    }

    private fun findPossibleLicensePlates(text: Text): List<Pair<String, Float>> {
        val possiblePlates = mutableListOf<Pair<String, Float>>()

        // 전체 텍스트에서 클린 텍스트 생성
        val fullText = text.text
        val cleanText = fullText.replace("\\s+".toRegex(), "")

        // 패턴 리스트에서 매칭 시도
        val patternMatches = mutableListOf<String>()

        // 원본 텍스트에서 검색
        LICENSE_PATTERNS.forEach { pattern ->
            val matcher = pattern.matcher(fullText)
            while (matcher.find()) {
                patternMatches.add(matcher.group())
            }
        }

        // 공백 제거 텍스트에서 검색
        LICENSE_PATTERNS.forEach { pattern ->
            val matcher = pattern.matcher(cleanText)
            while (matcher.find()) {
                val match = matcher.group()
                if (!patternMatches.contains(match)) {
                    patternMatches.add(match)
                }
            }
        }

        // 블록별 세부 분석
        for (block in text.textBlocks) {
            for (line in block.lines) {
                val lineText = line.text.replace("\\s+".toRegex(), "")

                LICENSE_PATTERNS.forEach { pattern ->
                    val matcher = pattern.matcher(lineText)
                    if (matcher.find()) {
                        val match = matcher.group()
                        // 글자별 신뢰도 확인 (TextBlock API에서 지원하는 경우)
                        var confidence = calculateConfidence(line)

                        // 추가 검증 (번호판 형식 검증)
                        val validatedPlate = validateAndCorrectPlate(match)
                        if (validatedPlate.isNotEmpty()) {
                            possiblePlates.add(Pair(validatedPlate, confidence))
                        }
                    }
                }
            }
        }

        // 중복 제거 및 신뢰도 기준 정렬
        return possiblePlates.distinctBy { it.first }
    }

    private fun calculateConfidence(line: Text.Line): Float {
        // ML Kit에서 confidence를 제공하지 않는 경우 휴리스틱 사용
        // 1. 텍스트 선명도 - 경계 상자 크기와 문자 수 비율
        val boundingBox = line.boundingBox ?: return 0.5f // 기본값
        val textLength = line.text.replace("\\s+".toRegex(), "").length

        // 번호판 문자는 일정한 간격으로 배치됨, 그 비율 체크
        val width = boundingBox.width()
        val height = boundingBox.height()

        // 일반적인 번호판 비율(가로:세로) 확인 (한국 번호판 기준 약 4.3:1)
        val aspectRatio = width.toFloat() / height.toFloat()
        val aspectConfidence = if (aspectRatio in 3.5f..5.0f) 0.3f else 0.1f

        // 문자 밀도 확인 (번호판 문자는 균등 간격)
        val charDensity = textLength.toFloat() / width
        val densityConfidence = if (charDensity in 0.05f..0.15f) 0.3f else 0.1f

        // 문자열 패턴 강도 확인
        val patternConfidence = 0.4f

        return aspectConfidence + densityConfidence + patternConfidence
    }

    private fun validateAndCorrectPlate(plate: String): String {
        // 번호판 형식 검증 및 오류 수정
        val corrected = plate.trim()
            .replace("O", "0") // 'O'를 '0'으로 교체
            .replace("I", "1") // 'I'를 '1'로 교체
            .replace("B", "8") // 가능한 B와 8 혼동 수정

        // 지역명 검증
        val regionCodes = listOf("서울", "경기", "인천", "강원", "충북", "충남", "대전", "경북", "경남", "부산", "울산", "대구", "전북", "전남", "광주", "제주")

        // 두 가지 패턴 검증
        if (NEW_CAR_PATTERN.matcher(corrected).matches()) {
            return corrected
        } else if (BUSINESS_LICENSE_PATTERN.matcher(corrected).matches()) {
            // 사업용 번호판 첫 두 글자가 지역명인지 확인
            val region = corrected.substring(0, 2)
            if (regionCodes.contains(region)) {
                return corrected
            }
        } else if (OLD_CAR_PATTERN.matcher(corrected).matches()) {
            return corrected
        }

        return ""  // 검증 실패
    }

    private fun findPlateRegion(text: Text): Rect? {
        var bestRect: Rect? = null
        var highestConfidence = 0f

        for (block in text.textBlocks) {
            for (line in block.lines) {
                val lineText = line.text.replace("\\s+".toRegex(), "")

                LICENSE_PATTERNS.forEach { pattern ->
                    if (pattern.matcher(lineText).find()) {
                        val confidence = calculateConfidence(line)
                        if (confidence > highestConfidence) {
                            highestConfidence = confidence
                            bestRect = line.boundingBox
                        }
                    }
                }
            }
        }

        // 번호판 영역을 약간 확장 (더 넓은 컨텍스트 포함)
        bestRect?.let {
            val expandedRect = Rect(
                it.left - (it.width() * 0.1).toInt().coerceAtLeast(0),
                it.top - (it.height() * 0.2).toInt().coerceAtLeast(0),
                it.right + (it.width() * 0.1).toInt(),
                it.bottom + (it.height() * 0.2).toInt()
            )
            return expandedRect
        }

        return bestRect
    }

    private fun sendTextOnly(licensePlate: String) {
        if (licensePlate.isEmpty()) {
            serverStatusListener("전송 실패: 번호판 텍스트 없음")
            return
        }

        val jsonObject = JSONObject().apply {
            put("licensePlate", licensePlate)
            put("timestamp", System.currentTimeMillis())
            put("deviceId", android.os.Build.MODEL)
        }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("data", jsonObject.toString())
            .build()

        sendRequest(requestBody)
    }

    private fun sendFullFrame(licensePlate: String, imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap() ?: return
        sendWithImage(licensePlate, bitmap, "$licensePlate.jpg")
    }

    private fun sendCroppedPlateImage(licensePlate: String, text: Text, imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap() ?: return
        val plateRegion = findPlateRegion(text)

        if (plateRegion != null) {
            // 확인된 영역이 이미지 경계 내에 있는지 확인
            val left = plateRegion.left.coerceAtLeast(0)
            val top = plateRegion.top.coerceAtLeast(0)
            val width = plateRegion.width().coerceAtMost(bitmap.width - left)
            val height = plateRegion.height().coerceAtMost(bitmap.height - top)

            if (width > 0 && height > 0) {
                val croppedBitmap = Bitmap.createBitmap(bitmap, left, top, width, height)
                sendWithImage(licensePlate, croppedBitmap, "cropped_plate.jpg")
            } else {
                // 크롭 영역이 유효하지 않으면 전체 이미지 전송
                sendWithImage(licensePlate, bitmap, "$licensePlate.jpg")
            }
        } else {
            // 번호판 영역을 찾지 못하면 전체 이미지 전송
            sendWithImage(licensePlate, bitmap, "$licensePlate.jpg")
        }
    }

    private fun sendWithImage(licensePlate: String, imageBitmap: Bitmap, fileName: String) {
        if (licensePlate.isEmpty()) {
            serverStatusListener("전송 실패: 번호판 텍스트 없음")
            return
        }

        // 텍스트 데이터 JSON 생성
        val jsonObject = JSONObject().apply {
            put("licensePlate", licensePlate)
            put("timestamp", System.currentTimeMillis())
            put("deviceId", android.os.Build.MODEL)
            put("confidence", recentDetections[licensePlate] ?: 1)
        }

        // 이미지 데이터 바이트 배열로 변환
        val byteArrayOutputStream = ByteArrayOutputStream()
        imageBitmap.compress(Bitmap.CompressFormat.JPEG, 80, byteArrayOutputStream)
        val imageData = byteArrayOutputStream.toByteArray()

        // MultipartBody 생성 - 텍스트와 이미지 데이터를 함께 보내기
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("data", jsonObject.toString())  // 텍스트 데이터
            .addFormDataPart("image", fileName, RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageData))  // 이미지 데이터
            .build()

        // 서버로 요청 보내기
        sendRequest(requestBody)
    }

    private fun sendRequest(requestBody: RequestBody) {
        val request = Request.Builder()
            .url(SERVER_URL)
            .post(requestBody)
            .build()

        serverStatusListener("서버로 전송 중...")

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                serverStatusListener("전송 실패: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    serverStatusListener("전송 성공")
                } else {
                    serverStatusListener("전송 실패: 서버 오류 ${response.code}")
                }
            }
        })
    }

    companion object {
        private const val TAG = "ImprovedLicensePlateProcessor"
        private const val SERVER_URL = "https://admin.campingfriend.co.kr/api/license"

        private const val IMAGE_SEND_MODE = "FULL_FRAME" // 전송 모드: "NONE", "FULL_FRAME", "CROPPED_PLATE"
        private const val DETECTION_COOLDOWN_MS = 3000L
        private const val SERVER_SEND_COOLDOWN_MS = 5000L

        // 다양한 번호판 패턴 정의
        private val OLD_CAR_PATTERN = Pattern.compile("\\d{2,3}[가-힣]\\d{4}")         // 12가1234
        private val NEW_CAR_PATTERN = Pattern.compile("\\d{2,3}[가-힣]\\d{4}")         // 123가1234
        private val BUSINESS_LICENSE_PATTERN = Pattern.compile("[가-힣]{2}\\d{2}[가-힣]\\d{4}")  // 서울12가1234
        private val RENTAL_CAR_PATTERN = Pattern.compile("\\d{2,3}[하-힣]\\d{4}")      // 렌터카 번호판
        private val TAXI_PATTERN = Pattern.compile("\\d{2,3}[바-사]\\d{4}")           // 택시 번호판
        private val DIPLOMATIC_PATTERN = Pattern.compile("\\d{2,3}[아-자]\\d{4}")      // 외교 번호판
        private val TEMPORARY_PATTERN = Pattern.compile("\\d{2,3}[파-하]\\d{4}")      // 임시 번호판

        // 모든 패턴 리스트
        private val LICENSE_PATTERNS = listOf(
            OLD_CAR_PATTERN,
            NEW_CAR_PATTERN,
            BUSINESS_LICENSE_PATTERN,
            RENTAL_CAR_PATTERN,
            TAXI_PATTERN,
            DIPLOMATIC_PATTERN,
            TEMPORARY_PATTERN
        )
    }
}

// 이미지 처리를 위한 클래스 (아직 구현 필요)
object ImageProcessor {
    fun process(image: Image, rotationDegrees: Int): Image {
        // 여기에 이미지 전처리 로직 구현
        // - 대비 향상
        // - 노이즈 감소
        // - 이진화(흑백화) 등

        // 예제 코드이므로 실제 구현은 생략하고 원본 이미지 반환
        return image
    }
}



// 이미지 변환 확장 함수
fun ImageProxy.toBitmap(): Bitmap? {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}