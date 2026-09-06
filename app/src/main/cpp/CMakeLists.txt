cmake_minimum_required(VERSION 3.22.1)
project("magicimagepro")

if(NOT OpenCV_DIR AND DEFINED ENV{OPENCV_DIR})
        set(OpenCV_DIR "$ENV{OPENCV_DIR}")
endif()

if(NOT OpenCV_DIR)
        set(local_opencv_dir "${CMAKE_CURRENT_LIST_DIR}/../../../../../../OpenCV-android-sdk/sdk/native/jni")
        if(EXISTS "${local_opencv_dir}/OpenCVConfig.cmake")
                set(OpenCV_DIR "${local_opencv_dir}")
        endif()
endif()

find_package(OpenCV REQUIRED)

add_library(
        removal_engine
        SHARED
        native-lib.cpp
)

# Link OpenCV libraries, Android log, and graphics buffer libraries
target_link_libraries(
        removal_engine
        ${OpenCV_LIBS}
        log
        jnigraphics
)
