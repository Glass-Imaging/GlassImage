cmake_minimum_required(VERSION 3.22)

# Features
include(FetchContent)

# Project
project("glass_image")

# This will make cmake generate compile_commands.json, used by clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(GLASS_IMAGE_BUILD_IMAGE_IO "Build with image I/O libraries." OFF)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE) # TODO: Needed for something?

option(GLASS_IMAGE_RUN_ASSERTIONS "Run assertions, independently of the release/debug build type." OFF)
option(GLASS_IMAGE_PAD_POWER2_IMAGES "Pad GpuImages with a power of 2 row pitch to a non-power-2 for speedup." ON)

# Compile options
if(GLASS_IMAGE_RUN_ASSERTIONS)
    add_compile_options(-UNDEBUG)
endif()
if(GLASS_IMAGE_PAD_POWER2_IMAGES)
    add_compile_definitions(GLASS_IMAGE_PAD_POWER2_IMAGES)
endif()

# Add dependencies
if(NOT TARGET GlassLogging)
    add_subdirectory(dependencies/internal/GlassLogging)
endif()

# Directories
add_subdirectory(src)

# Add unit tests
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)
add_subdirectory(tests/unit)