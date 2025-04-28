cmake_minimum_required(VERSION 3.22)

project("glass_image")
# Options
option(GLASS_IMAGE_BUILD_IMAGE_IO "Build with image I/O libraries." OFF)

add_subdirectory(src)