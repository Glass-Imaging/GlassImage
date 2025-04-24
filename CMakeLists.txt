# Build target for GlassImage.
## Building this standalone is not implemented yet.

add_library(
    GlassImage
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_cl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_cl_error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_color_science.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_icd_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_ocl.cpp

    $<$<BOOL:${DEFINE_ANDROID_TOOLCHAIN}>:${CMAKE_CURRENT_SOURCE_DIR}/src/gls_android_support.cpp>
)

target_include_directories(
    GlassImage
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/android/support_libs/headers
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenCLHeaders
)

# Possibly build with image I/O libraries.
if(BUILD_IMAGE_IO)
    message("Building with Image IO Libraries")

    # Add Homebrew zlib library path for macOS
    if(APPLE)
        set(ZLIB_ROOT /opt/homebrew/opt/zlib)
        set(PNG_ROOT /opt/homebrew/opt/libpng)
        set(JPEG_ROOT /opt/homebrew/opt/jpeg)
        set(TIFF_ROOT /opt/homebrew/opt/libtiff)
    endif()

    # Find packages
    find_package(ZLIB REQUIRED)
    find_package(PNG REQUIRED)
    find_package(JPEG REQUIRED)
    find_package(TIFF REQUIRED)    

    target_sources(
        GlassImage
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_image_png.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_dng_lossless_jpeg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_image_jpeg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_image_tiff.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gls_tiff_metadata.cpp
    )

    target_link_libraries(
        GlassImage
        ZLIB::ZLIB
        PNG::PNG
        JPEG::JPEG
        TIFF::TIFF
    )
endif()