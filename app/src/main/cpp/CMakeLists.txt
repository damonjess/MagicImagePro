cmake_minimum_required(VERSION 3.22.1)
project("magicimagepro")

# 1. Tell CMake where to find the OpenCV C++ headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opencv_includes)

# 2. Import the pre-built OpenCV shared library
add_library(lib_opencv SHARED IMPORTED)

# 3. Point CMake to the specific .so file for the active architecture (e.g., arm64-v8a)
# Note: If you downloaded OpenCV 3.x, change this to libopencv_java3.so
set_target_properties(lib_opencv PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java4.so
)

# Your custom C++ engine
add_library(
        removal_engine
        SHARED
        native-lib.cpp)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# 4. Link OpenCV alongside your other libraries
target_link_libraries(
        removal_engine
        lib_opencv
        ${log-lib}
        ${jnigraphics-lib})
