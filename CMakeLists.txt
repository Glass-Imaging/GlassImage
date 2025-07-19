cmake_minimum_required(VERSION 3.22)

project("glass_image")

# This will make cmake generate compile_commands.json, used by clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(GLASS_IMAGE_BUILD_IMAGE_IO "Build with image I/O libraries." OFF)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE) # TODO: Needed for something?

# Add dependencies
if(NOT TARGET GlassLogging)
    add_subdirectory(dependencies/internal/GlassLogging)
endif()

# Add sources
add_subdirectory(src)
