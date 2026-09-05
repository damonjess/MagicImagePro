# Fix missing libtensorflowlite.so build error

The build error is caused by `CMakeLists.txt` explicitly looking for `libtensorflowlite.so` in `src/main/jniLibs/arm64-v8a/`, which is currently empty. Instead of manually managing native binaries, we will use the **Prefab** feature of the Android Gradle Plugin to automatically link against the TensorFlow Lite library provided by the Gradle dependency.

## Proposed Changes

### [app]

#### [MODIFY] [build.gradle.kts](file:///home/damon/StudioProjects/MagicImagePro/app/build.gradle.kts)
- Enable the `prefab` build feature.

#### [MODIFY] [CMakeLists.txt](file:///home/damon/StudioProjects/MagicImagePro/app/src/main/cpp/CMakeLists.txt)
- Remove the manual `add_library(tensorflowlite SHARED IMPORTED)` and its `IMPORTED_LOCATION` property.
- Use `find_package(tensorflowlite REQUIRED CONFIG)` to locate the library via Prefab.
- Update `target_link_libraries` to link against `tensorflowlite::tensorflowlite`.

## Verification Plan

### Automated Tests
- Run `./gradlew :app:buildCMakeDebug[arm64-v8a]` to verify that the native build completes successfully.
- Run a full build `./gradlew :app:assembleDebug` to ensure everything is packaged correctly.
