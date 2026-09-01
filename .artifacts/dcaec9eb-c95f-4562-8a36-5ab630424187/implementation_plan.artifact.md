# Fix Build Dependencies and ML Model Initialization

The objective is to resolve the Gradle dependency issues and fix the `TFLiteModel` initialization to ensure the app builds and runs correctly with the Float32 models.

## Proposed Changes

### Build Configuration

#### [MODIFY] [libs.versions.toml](file:///C:/Users/Damon/AndroidStudioProjects/MagicImagePro/gradle/libs.versions.toml)
- Simplify library keys to avoid nested accessor issues in Gradle.
- Update TensorFlow Lite versions to 2.17.0 and Support to 0.5.0 for better compatibility.

#### [MODIFY] [app/build.gradle.kts](file:///C:/Users/Damon/AndroidStudioProjects/MagicImagePro/app/build.gradle.kts)
- Update dependency references to match the simplified keys in `libs.versions.toml`.

### ML Components

#### [MODIFY] [TFLiteModel.kt](file:///C:/Users/Damon/AndroidStudioProjects/MagicImagePro/app/src/main/java/com/example/magicimagepro/ml/TFLiteModel.kt)
- Remove unnecessary non-null assertion as `tempInterpreter` is guaranteed to be non-null after the CPU fallback block.

## Verification Plan

### Automated Tests
- Run `./gradlew assembleDebug` to verify the build completes successfully.

### Manual Verification
- Deploy the app to a device/emulator.
- Load an image.
- Perform "Mask" and "Remove" operations to verify the Object Removal flow.
- Perform "Upscale" to verify the ESRGAN flow.
