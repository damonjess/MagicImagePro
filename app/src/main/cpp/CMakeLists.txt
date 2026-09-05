cmake_minimum_required(VERSION 3.22.1)
project("magicimagepro")

# OpenCV_DIR is passed via Gradle arguments
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
