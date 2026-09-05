import java.net.URI

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.magicimagepro"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.magicimagepro"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += ""
                arguments += "-DOpenCV_DIR=/home/damon/OpenCV-android-sdk/sdk/native/jni"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        viewBinding = true
        prefab = true
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.google.material)
    implementation(libs.androidx.constraintlayout)

    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.support)
    implementation(libs.tensorflow.lite.gpu)

    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("com.alibaba.android:mnn:0.0.8")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}

val copyTfliteSo by tasks.registering(Copy::class) {
    val cfg = configurations.detachedConfiguration(dependencies.create("org.tensorflow:tensorflow-lite:2.17.0"))
    duplicatesStrategy = DuplicatesStrategy.WARN
    from({
        cfg.files.map { zipTree(it) }
    }) {
        include("jni/**/libtensorflowlite_jni.so", "jni/**/libtensorflowlite.so")
        eachFile {
            val abi = file.parentFile.name
            path = "$abi/libtensorflowlite.so"
        }
        includeEmptyDirs = false
    }
    into(layout.projectDirectory.dir("src/main/jniLibs"))
}

val downloadTfliteHeaders by tasks.registering {
    val destDir = layout.projectDirectory.dir("src/main/cpp/include")
    outputs.dir(destDir)
    doLast {
        val baseUrl = "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.17.0/"
        val relativePaths = listOf(
            "tensorflow/lite/c/c_api.h",
            "tensorflow/lite/c/c_api_types.h",
            "tensorflow/lite/c/common.h",
            "tensorflow/lite/core/c/c_api.h",
            "tensorflow/lite/core/c/c_api_types.h",
            "tensorflow/lite/core/c/common.h",
            "tensorflow/lite/builtin_ops.h"
        )
        relativePaths.forEach { relPath ->
            val file = destDir.file(relPath).asFile
            if (!file.exists()) {
                file.parentFile.mkdirs()
                println("Downloading $relPath...")
                val bytes = URI(baseUrl + relPath).toURL().readBytes()
                file.writeBytes(bytes)
            }
        }
    }
}

tasks.named("preBuild") {
    dependsOn(copyTfliteSo)
    dependsOn(downloadTfliteHeaders)
}
