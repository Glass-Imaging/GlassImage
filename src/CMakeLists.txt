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

target_include_directories( GlassImage SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../android/support_libs/headers )
target_include_directories( GlassImage SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../OpenCLHeaders )

target_include_directories(
    GlassImage
    PUBLIC
    ../include
    # ../android/support_libs/headers
    # ../OpenCLHeaders
)

# Possibly build with image I/O libraries.
if(GLASS_IMAGE_BUILD_IMAGE_IO AND (CMAKE_SYSTEM_NAME STREQUAL "Android"))
    message("Building with Image IO Libraries for Android")

    target_sources(
        GlassImage
        PRIVATE
        gls_image_png.cpp
        gls_dng_lossless_jpeg.cpp
        gls_image_jpeg.cpp
        gls_image_tiff.cpp
        gls_tiff_metadata.cpp
    )    

    add_library( libjpg STATIC IMPORTED )
    set_target_properties( libjpg PROPERTIES IMPORTED_LOCATION ${GLASS_IMAGE_PATH}/android/support_libs/jni/${ANDROID_ABI}/libjpeg-turbo.a )

    target_link_libraries(
        GlassImage
        ${CMAKE_CURRENT_SOURCE_DIR}/../android/support_libs/jni/${ANDROID_ABI}/libz.a
        ${CMAKE_CURRENT_SOURCE_DIR}/../android/support_libs/jni/${ANDROID_ABI}/libpng.a
        ${CMAKE_CURRENT_SOURCE_DIR}/../android/support_libs/jni/${ANDROID_ABI}/libjpeg-turbo.a
        ${CMAKE_CURRENT_SOURCE_DIR}/../android/support_libs/jni/${ANDROID_ABI}/libtiff.a
        ${CMAKE_CURRENT_SOURCE_DIR}/../android/support_libs/jni/${ANDROID_ABI}/libtiffxx.a
    )    

elseif(GLASS_IMAGE_BUILD_IMAGE_IO AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message("Building with Image IO Libraries for Darwin")

    # Add Homebrew zlib library path for macOS
    set(ZLIB_ROOT /opt/homebrew/opt/zlib)
    set(PNG_ROOT /opt/homebrew/opt/libpng)
    set(JPEG_ROOT /opt/homebrew/opt/jpeg)
    set(TIFF_ROOT /opt/homebrew/opt/libtiff)

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