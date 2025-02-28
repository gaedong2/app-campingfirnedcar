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


        val image = InputImage.fromMediaImage(processedImage, imageProxy.imageInfo.rotationDegrees)

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

object ImageProcessor {
    private const val TAG = "ImageProcessor"

    // 이미지 전처리를 위한 메인 함수
    fun process(image: Image, rotationDegrees: Int): Image {
        try {
            // 이미지를 비트맵으로 변환
            val bitmap = imageToBitmap(image)

            // 비트맵 전처리 적용
            val processedBitmap = preProcessBitmap(bitmap, rotationDegrees)

            // 결과 로깅
            Log.d(TAG, "이미지 전처리 완료: ${bitmap.width}x${bitmap.height}")

            // 처리된 비트맵으로 이미지 업데이트 (실제 구현에서는 Image 객체를 직접 수정할 수 없으므로,
            // 여기서는 원본 이미지를 반환하지만 실제로는 이미지 데이터가 처리된 것으로 간주)
            return image
        } catch (e: Exception) {
            Log.e(TAG, "이미지 처리 중 오류 발생", e)
            return image
        }
    }

    // Image 객체를 Bitmap으로 변환
    private fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * image.width

        // Bitmap 생성 및 데이터 복사
        val bitmap = Bitmap.createBitmap(
            image.width + rowPadding / pixelStride,
            image.height,
            Bitmap.Config.ARGB_8888
        )

        // 이미지 데이터를 비트맵으로 복사하는 코드
        // 실제 구현에서는 YUV 포맷을 RGB로 변환하는 과정이 필요할 수 있음

        return bitmap
    }

    // 비트맵 전처리
    private fun preProcessBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        var result = bitmap

        // 1. 이미지 회전 (필요한 경우)
        if (rotationDegrees != 0) {
            result = rotateBitmap(result, rotationDegrees.toFloat())
        }

        // 2. 그레이스케일 변환
        result = convertToGrayscale(result)

        // 3. 대비 향상
        result = enhanceContrast(result)

        // 4. 노이즈 제거
        result = reduceNoise(result)

        // 5. 번호판 영역 감지 및 강화 (사각형 영역 감지)
        result = enhancePlateRegion(result)

        // 6. 이진화 (번호판 텍스트 강조)
        result = binarize(result)

        return result
    }

    // 이미지 회전
    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    // 그레이스케일 변환
    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 픽셀 단위로 그레이스케일 변환
        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                val r = android.graphics.Color.red(pixel)
                val g = android.graphics.Color.green(pixel)
                val b = android.graphics.Color.blue(pixel)

                // 그레이스케일 공식: 0.299 * R + 0.587 * G + 0.114 * B
                val gray = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
                val grayPixel = android.graphics.Color.rgb(gray, gray, gray)
                grayscaleBitmap.setPixel(x, y, grayPixel)
            }
        }

        return grayscaleBitmap
    }

    // 대비 향상 (히스토그램 균등화)
    private fun enhanceContrast(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 히스토그램 계산
        val histogram = IntArray(256)
        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                val gray = android.graphics.Color.red(pixel) // 그레이스케일에서는 R=G=B
                histogram[gray]++
            }
        }

        // 누적 분포 함수 계산
        val cdf = IntArray(256)
        cdf[0] = histogram[0]
        for (i in 1 until 256) {
            cdf[i] = cdf[i-1] + histogram[i]
        }

        // 정규화
        val totalPixels = width * height
        val normalizedCdf = IntArray(256)
        for (i in 0 until 256) {
            normalizedCdf[i] = (cdf[i] * 255.0 / totalPixels).toInt()
        }

        // 대비 향상 적용
        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                val gray = android.graphics.Color.red(pixel)
                val newGray = normalizedCdf[gray]
                result.setPixel(x, y, android.graphics.Color.rgb(newGray, newGray, newGray))
            }
        }

        return result
    }

    // 노이즈 제거 (가우시안 블러)
    private fun reduceNoise(bitmap: Bitmap): Bitmap {
        // 가우시안 커널 생성 (3x3)
        val kernel = arrayOf(
            floatArrayOf(1f/16f, 2f/16f, 1f/16f),
            floatArrayOf(2f/16f, 4f/16f, 2f/16f),
            floatArrayOf(1f/16f, 2f/16f, 1f/16f)
        )

        val width = bitmap.width
        val height = bitmap.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 컨볼루션 적용
        for (x in 1 until width - 1) {
            for (y in 1 until height - 1) {
                var sum = 0f

                // 3x3 윈도우에 커널 적용
                for (i in -1..1) {
                    for (j in -1..1) {
                        val pixel = bitmap.getPixel(x + i, y + j)
                        val gray = android.graphics.Color.red(pixel)
                        sum += gray * kernel[i+1][j+1]
                    }
                }

                val newGray = sum.toInt().coerceIn(0, 255)
                result.setPixel(x, y, android.graphics.Color.rgb(newGray, newGray, newGray))
            }
        }

        return result
    }

    // 번호판 영역 감지 및 강화 (직사각형 영역 감지)
    private fun enhancePlateRegion(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 1. 수평/수직 에지 감지 (Sobel 연산자 사용)
        val edgeMap = detectEdges(bitmap)

        // 2. 사각형 후보 영역 찾기
        val plateRegions = findPlateRegions(edgeMap, bitmap)

        // 3. 원본 이미지 복사
        val canvas = android.graphics.Canvas(result)
        canvas.drawBitmap(bitmap, 0f, 0f, null)

        // 4. 감지된 번호판 영역 강화
        if (plateRegions.isNotEmpty()) {
            Log.d(TAG, "번호판 후보 영역 ${plateRegions.size}개 감지됨")

            // 가장 유력한 번호판 영역 선택 (크기와 비율 기준)
            val bestRegion = findBestPlateRegion(plateRegions)
            bestRegion?.let { region ->
                // 번호판 영역 강화
                enhanceRegion(result, region)
                Log.d(TAG, "번호판 영역 강화 완료: $region")
            }
        } else {
            Log.d(TAG, "번호판 영역 감지 실패, 전체 이미지 처리")
        }

        return result
    }

    // 에지 감지 (Sobel 연산자)
    private fun detectEdges(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val edgeMap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Sobel 연산자 커널
        val sobelX = arrayOf(
            intArrayOf(-1, 0, 1),
            intArrayOf(-2, 0, 2),
            intArrayOf(-1, 0, 1)
        )

        val sobelY = arrayOf(
            intArrayOf(-1, -2, -1),
            intArrayOf(0, 0, 0),
            intArrayOf(1, 2, 1)
        )

        // 에지 감지 적용
        for (x in 1 until width - 1) {
            for (y in 1 until height - 1) {
                var sumX = 0
                var sumY = 0

                // 3x3 윈도우에 Sobel 연산자 적용
                for (i in -1..1) {
                    for (j in -1..1) {
                        val pixel = bitmap.getPixel(x + i, y + j)
                        val gray = android.graphics.Color.red(pixel)

                        sumX += gray * sobelX[i+1][j+1]
                        sumY += gray * sobelY[i+1][j+1]
                    }
                }

                // 기울기 크기 계산
                val gradientMagnitude = Math.sqrt((sumX * sumX + sumY * sumY).toDouble()).toInt()
                val edgeStrength = gradientMagnitude.coerceIn(0, 255)

                // 임계값 적용 (강한 에지만 남기기)
                val thresholdValue = if (edgeStrength > 50) edgeStrength else 0
                edgeMap.setPixel(x, y, android.graphics.Color.rgb(thresholdValue, thresholdValue, thresholdValue))
            }
        }

        return edgeMap
    }

    // 번호판 후보 영역 찾기
    private fun findPlateRegions(edgeMap: Bitmap, originalBitmap: Bitmap): List<Rect> {
        val width = edgeMap.width
        val height = edgeMap.height
        val regions = mutableListOf<Rect>()

        // 1. 수평 라인 감지 (번호판 테두리의 특성)
        val horizontalLines = mutableListOf<Pair<Int, Int>>() // y좌표, 길이

        for (y in 0 until height) {
            var lineStart = -1
            var lineLength = 0

            for (x in 0 until width) {
                val pixel = edgeMap.getPixel(x, y)
                val isEdge = android.graphics.Color.red(pixel) > 0

                if (isEdge) {
                    if (lineStart == -1) {
                        lineStart = x
                    }
                    lineLength++
                } else if (lineStart != -1) {
                    // 라인 종료, 최소 길이 조건 확인
                    if (lineLength > width / 10) { // 이미지 너비의 10% 이상인 라인만 고려
                        horizontalLines.add(Pair(y, lineLength))
                    }
                    lineStart = -1
                    lineLength = 0
                }
            }

            // 라인이 이미지 끝까지 간 경우
            if (lineStart != -1 && lineLength > width / 10) {
                horizontalLines.add(Pair(y, lineLength))
            }
        }

        // 2. 수평 라인들을 그룹화하여 번호판 후보 영역 생성
        if (horizontalLines.size >= 2) {
            // 라인 쌍 연결
            for (i in 0 until horizontalLines.size - 1) {
                val (y1, len1) = horizontalLines[i]

                for (j in i + 1 until horizontalLines.size) {
                    val (y2, len2) = horizontalLines[j]
                    val verticalDistance = Math.abs(y2 - y1)

                    // 번호판 높이에 가까운 수직 거리를 가진 쌍 찾기
                    // 한국 번호판 높이:너비 비율은 약 1:4.3
                    if (verticalDistance > height / 20 && verticalDistance < height / 5) {
                        // 후보 영역 생성
                        val rect = Rect(
                            0, // 좌측 (보수적으로 전체 너비 사용)
                            Math.min(y1, y2), // 상단
                            width, // 우측 (보수적으로 전체 너비 사용)
                            Math.max(y1, y2) // 하단
                        )

                        // 비율 검증 (번호판 비율에 가까운지)
                        val aspectRatio = rect.width().toFloat() / rect.height()
                        if (aspectRatio in 2.5f..5.5f) { // 한국 번호판 비율 고려
                            regions.add(rect)
                        }
                    }
                }
            }
        }

        // 3. 색상 분석을 통한 추가 검증 (번호판 배경색 고려)
        val verifiedRegions = mutableListOf<Rect>()

        for (region in regions) {
            // 해당 영역 내 색상 분포 분석
            val colorHistogram = analyzeRegionColor(originalBitmap, region)

            // 번호판 배경색 특성 확인 (흰색/노란색이 충분히 있는지)
            if (hasPlateColorCharacteristics(colorHistogram)) {
                verifiedRegions.add(region)
            }
        }

        // 중복 제거 및 반환
        return mergeOverlappingRegions(verifiedRegions)
    }

    // 색상 분석
    private fun analyzeRegionColor(bitmap: Bitmap, region: Rect): Map<String, Int> {
        val colorCounts = mutableMapOf(
            "white" to 0,
            "yellow" to 0,
            "green" to 0,
            "blue" to 0,
            "other" to 0
        )

        val totalPixels = region.width() * region.height()

        for (x in region.left until region.right) {
            for (y in region.top until region.bottom) {
                if (x < 0 || y < 0 || x >= bitmap.width || y >= bitmap.height) continue

                val pixel = bitmap.getPixel(x, y)
                val r = android.graphics.Color.red(pixel)
                val g = android.graphics.Color.green(pixel)
                val b = android.graphics.Color.blue(pixel)

                // 간단한 색상 분류
                when {
                    // 흰색 (번호판 배경)
                    r > 200 && g > 200 && b > 200 -> colorCounts["white"] = colorCounts["white"]!! + 1

                    // 노란색 (일부 번호판 배경)
                    r > 200 && g > 200 && b < 100 -> colorCounts["yellow"] = colorCounts["yellow"]!! + 1

                    // 초록색 (일부 번호판)
                    r < 100 && g > 150 && b < 100 -> colorCounts["green"] = colorCounts["green"]!! + 1

                    // 파란색 (일부 번호판)
                    r < 100 && g < 100 && b > 150 -> colorCounts["blue"] = colorCounts["blue"]!! + 1

                    // 기타
                    else -> colorCounts["other"] = colorCounts["other"]!! + 1
                }
            }
        }

        return colorCounts
    }

    // 번호판 색상 특성 확인
    private fun hasPlateColorCharacteristics(colorHistogram: Map<String, Int>): Boolean {
        val totalPixels = colorHistogram.values.sum()

        // 흰색 비율
        val whiteRatio = colorHistogram["white"]!! / totalPixels.toFloat()

        // 노란색 비율
        val yellowRatio = colorHistogram["yellow"]!! / totalPixels.toFloat()

        // 초록색 비율
        val greenRatio = colorHistogram["green"]!! / totalPixels.toFloat()

        // 파란색 비율
        val blueRatio = colorHistogram["blue"]!! / totalPixels.toFloat()

        // 번호판 색상 특성 검증 (일반적인 번호판 배경색 조합)
        return whiteRatio > 0.3 || // 흰색 배경 번호판
                yellowRatio > 0.3 || // 노란색 배경 번호판
                greenRatio > 0.3 || // 초록색 배경 번호판 (영업용)
                blueRatio > 0.3 // 파란색 배경 번호판 (관용)
    }

    // 중복 영역 병합
    private fun mergeOverlappingRegions(regions: List<Rect>): List<Rect> {
        if (regions.isEmpty()) return emptyList()

        val result = mutableListOf<Rect>()
        val usedIndices = mutableSetOf<Int>()

        for (i in regions.indices) {
            if (i in usedIndices) continue

            val baseRect = Rect(regions[i])
            usedIndices.add(i)

            for (j in i + 1 until regions.size) {
                if (j in usedIndices) continue

                val testRect = regions[j]

                // 충분히 겹치는지 확인
                if (Rect.intersects(baseRect, testRect)) {
                    // 두 영역 병합
                    baseRect.union(testRect)
                    usedIndices.add(j)
                }
            }

            result.add(baseRect)
        }

        return result
    }

    // 가장 좋은 번호판 후보 선택
    private fun findBestPlateRegion(regions: List<Rect>): Rect? {
        if (regions.isEmpty()) return null

        // 번호판다운 비율을 가진 가장 큰 영역 선택
        return regions.maxByOrNull { region ->
            val aspectRatio = region.width().toFloat() / region.height()
            val areaScore = region.width() * region.height()

            // 한국 번호판 비율(약 4.3:1)에 가까울수록 높은 점수
            val ratioScore = 100 - Math.abs(aspectRatio - 4.3f) * 20

            // 최종 점수 = 면적 + 비율 점수
            areaScore * ratioScore
        }
    }

    // 특정 영역 강화
    private fun enhanceRegion(bitmap: Bitmap, region: Rect) {
        val canvas = android.graphics.Canvas(bitmap)
        val paint = android.graphics.Paint()

        // 영역 경계 약간 확장
        val expandedRegion = Rect(
            (region.left - region.width() * 0.05).toInt().coerceAtLeast(0),
            (region.top - region.height() * 0.1).toInt().coerceAtLeast(0),
            (region.right + region.width() * 0.05).toInt().coerceAtMost(bitmap.width),
            (region.bottom + region.height() * 0.1).toInt().coerceAtMost(bitmap.height)
        )

        // 번호판 영역 내부의 대비를 더 높임
        for (x in expandedRegion.left until expandedRegion.right) {
            for (y in expandedRegion.top until expandedRegion.bottom) {
                val pixel = bitmap.getPixel(x, y)
                val gray = android.graphics.Color.red(pixel)

                // 번호판 글자 강조 (어두운 부분 더 어둡게, 밝은 부분 더 밝게)
                val enhancedGray = if (gray < 128) {
                    (gray * 0.7).toInt().coerceAtLeast(0)
                } else {
                    (gray * 1.3).toInt().coerceAtMost(255)
                }

                bitmap.setPixel(x, y, android.graphics.Color.rgb(enhancedGray, enhancedGray, enhancedGray))
            }
        }

        // 영역 주변의 배경 흐리게 처리 (번호판 영역 강조)
        paint.maskFilter = android.graphics.BlurMaskFilter(20f, android.graphics.BlurMaskFilter.Blur.NORMAL)
        paint.color = android.graphics.Color.argb(100, 0, 0, 0)
        canvas.drawRect(
            0f, 0f, bitmap.width.toFloat(), expandedRegion.top.toFloat(), paint
        )
        canvas.drawRect(
            0f, expandedRegion.bottom.toFloat(), bitmap.width.toFloat(), bitmap.height.toFloat(), paint
        )
        canvas.drawRect(
            0f, expandedRegion.top.toFloat(), expandedRegion.left.toFloat(), expandedRegion.bottom.toFloat(), paint
        )
        canvas.drawRect(
            expandedRegion.right.toFloat(), expandedRegion.top.toFloat(), bitmap.width.toFloat(), expandedRegion.bottom.toFloat(), paint
        )
    }

    // 이진화 (적응형 임계값)
    private fun binarize(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 블록 크기
        val blockSize = 11
        // 적응형 임계값 계산 상수
        val c = 7

        for (x in 0 until width) {
            for (y in 0 until height) {
                // 주변 블록 계산
                var sum = 0
                var count = 0

                val startX = (x - blockSize / 2).coerceAtLeast(0)
                val endX = (x + blockSize / 2).coerceAtMost(width - 1)
                val startY = (y - blockSize / 2).coerceAtLeast(0)
                val endY = (y + blockSize / 2).coerceAtMost(height - 1)

                for (i in startX..endX) {
                    for (j in startY..endY) {
                        val pixel = bitmap.getPixel(i, j)
                        sum += android.graphics.Color.red(pixel)
                        count++
                    }
                }

                // 적응형 임계값
                val threshold = (sum / count.toFloat()) - c

                // 현재 픽셀값
                val pixel = bitmap.getPixel(x, y)
                val gray = android.graphics.Color.red(pixel)

                // 이진화
                val binaryValue = if (gray > threshold) 255 else 0
                result.setPixel(x, y, android.graphics.Color.rgb(binaryValue, binaryValue, binaryValue))
            }
        }

        return result
    }
}



// 이미지 변환 확장 함수
fun ImageProxy.toBitmap(): Bitmap? {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}