add_library(
    GlassImage
    STATIC
    gls_cl.cpp
    gls_cl_error.cpp
    gls_color_science.cpp
    gls_icd_wrapper.cpp
    gls_logging.cpp
    gls_ocl.cpp

    # Only adding this if building for Android.
    $<$<BOOL:${DEFINE_ANDROID_TOOLCHAIN}>:/gls_android_support.cpp>
)

target_include_directories(
    GlassImage
    PUBLIC
    ../include
    ../android/support_libs/headers
    ../OpenCLHeaders
)

# Possibly build with image I/O libraries.
if(GLASS_IMAGE_BUILD_IMAGE_IO)
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
        gls_image_png.cpp
        gls_dng_lossless_jpeg.cpp
        gls_image_jpeg.cpp
        gls_image_tiff.cpp
        gls_tiff_metadata.cpp
    )

    target_link_libraries(
        GlassImage
        ZLIB::ZLIB
        PNG::PNG
        JPEG::JPEG
        TIFF::TIFF
    )
endif()

set_target_properties( 
    GlassImage
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)