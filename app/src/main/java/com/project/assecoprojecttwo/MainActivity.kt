package com.project.assecoprojecttwo

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    private val firebaseFaceDetector = FaceDetection.getClient(realTimeOpts)
    private lateinit var ivInputImage: ImageView
    private lateinit var txtOutputAdults: TextView
    private lateinit var txtOutputKids: TextView

    private val coroutineScope = CoroutineScope(Dispatchers.Main)
    private val requestImageCapture = 101
    private val requestImageSelect = 102
    private lateinit var photoPath: String
    private lateinit var ageInterpreter: Interpreter
    private lateinit var ageModel: AgeModel
    private val compatList = CompatibilityList()
    private var modelFilename = "model_age.tflite"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ivInputImage = findViewById(R.id.iv_input_image)

        txtOutputAdults = findViewById(R.id.txt_output_adults)
        txtOutputKids = findViewById(R.id.txt_output_kids)

        val options = Interpreter.Options().apply {
            addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
        }

        coroutineScope.launch {
            initModels(options)
        }
    }

    fun openCamera(v: View) {
        dispatchTakePictureIntent()
    }

    fun selectImage(v: View) {
        dispatchSelectPictureIntent()
    }

    private suspend fun initModels(options: Interpreter.Options) =
        withContext(Dispatchers.Default) {
            ageInterpreter =
                Interpreter(FileUtil.loadMappedFile(applicationContext, modelFilename), options)

            withContext(Dispatchers.Main) {
                ageModel = AgeModel().apply {
                    interpreter = ageInterpreter
                }
            }
        }

    override fun onDestroy() {
        super.onDestroy()
        ageInterpreter.close()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == requestImageCapture) {
            var bitmap = BitmapFactory.decodeFile(photoPath)
            val exifInterface = ExifInterface(photoPath)
            bitmap =
                when (exifInterface.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_UNDEFINED
                )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270f)
                    else -> bitmap
                }
            facesDetector(bitmap!!)
        } else if (resultCode == RESULT_OK && requestCode == requestImageSelect) {
            val inputStream = contentResolver.openInputStream(data?.data!!)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            facesDetector(bitmap!!)
        }
    }

    private fun facesDetector(image: Bitmap) {
        val inputImage = InputImage.fromBitmap(image, 0)
        var kid = 0
        var adult = 0

        firebaseFaceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.size != 0) {
                    if (faces.size > 1) {
                        Toast.makeText(this, "More than 1 face", Toast.LENGTH_SHORT).show()

                        coroutineScope.launch {
                            val age = ageModel.predictAge(image)
                            for (it in faces) {
                                if (age >= 18) {
                                    adult++
                                } else {
                                    kid++
                                }
                            }
                            txtOutputAdults.text = adult.toDouble().toInt().toString()
                            txtOutputKids.text = kid.toDouble().toInt().toString()
                        }
                    } else {
                        coroutineScope.launch {
                            val age = ageModel.predictAge(image)
                            if (age >= 18) {
                                adult++
                                txtOutputAdults.text = adult.toDouble().toInt().toString()
                            } else {
                                kid++
                                txtOutputKids.text = kid.toDouble().toInt().toString()
                            }
                        }
                    }
                    ivInputImage.setImageBitmap(image)
                } else {
                    val dialog = AlertDialog.Builder(this).apply {
                        title = "No Faces Found"
                        setMessage(
                            "Could not find any faces"
                        )
                        setPositiveButton("OK") { dialog, which ->
                            dialog.dismiss()
                        }
                        setCancelable(false)
                        create()
                    }
                    dialog.show()
                }
            }
        println(adult)
    }

    private fun createImageFile(): File {
        val imagesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("image", ".jpg", imagesDir).apply {
            photoPath = absolutePath
        }
    }

    private fun dispatchSelectPictureIntent() {
        val selectPictureIntent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "image/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        startActivityForResult(selectPictureIntent, requestImageSelect)
    }

    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(packageManager) != null) {
            val photoFile: File? = try {
                createImageFile()
            } catch (ex: IOException) {
                null
            }
            photoFile?.also {
                val photoURI = FileProvider.getUriForFile(
                    this,
                    "com.ml.projects.age_genderdetection", it
                )
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(takePictureIntent, requestImageCapture)
            }
        }
    }

    private fun rotateBitmap(original: Bitmap, degrees: Float): Bitmap? {
        val matrix = Matrix()
        matrix.preRotate(degrees)
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
    }
}