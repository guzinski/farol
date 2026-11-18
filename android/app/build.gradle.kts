plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

val commit = providers.exec {
    workingDir = rootDir
    commandLine("git", "rev-parse", "--short", "HEAD")
    isIgnoreExitValue = true
}.standardOutput.asText.map { it.trim().ifEmpty { "unknown" } }

android {
    namespace = "io.github.guzinski.farol"
    compileSdk = 37
    defaultConfig {
        applicationId = "io.github.guzinski.farol"
        minSdk = 35
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
        buildConfigField("String", "GIT_COMMIT", "\"${commit.get()}\"")
    }
    buildFeatures { compose = true; buildConfig = true }
    androidResources { noCompress += "tflite" }
}

dependencies {
    implementation(libs.activity.compose)
    implementation(libs.material3)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.video)
    implementation(libs.camera.view)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    testImplementation(libs.junit)
    testImplementation(libs.json)
}
