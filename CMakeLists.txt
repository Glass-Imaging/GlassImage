cmake_minimum_required(VERSION 3.22)

project("glass_image")

# Options
option(GLASS_IMAGE_BUILD_IMAGE_IO "Build with image I/O libraries." OFF)

add_compile_definitions("__ANDROID__")
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

add_subdirectory(src)