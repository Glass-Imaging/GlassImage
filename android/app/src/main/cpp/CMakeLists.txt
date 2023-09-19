# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("glassimagetest")

add_library( libz STATIC IMPORTED )
set_target_properties( libz PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jni/${ANDROID_ABI}/libz.a )
add_library( libpng STATIC IMPORTED )
set_target_properties( libpng PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jni/${ANDROID_ABI}/libpng.a )
add_library( libjpg STATIC IMPORTED )
set_target_properties( libjpg PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jni/${ANDROID_ABI}/libjpeg-turbo.a )

add_library( libtiff STATIC IMPORTED )
set_target_properties( libtiff PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jni/${ANDROID_ABI}/libtiff.a )
add_library( libtiffxx STATIC IMPORTED )
set_target_properties( libtiffxx PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/jni/${ANDROID_ABI}/libtiffxx.a )

add_executable( # Sets the name of the library.
        glassImageTest

        # Sets the library as a shared library.
        # SHARED

        # Provides a relative path to your source file(s).
        glassimage/src/gls_icd_wrapper.cpp
        glassimage/src/gls_cl_error.cpp
        glassimage/src/gls_logging.cpp
        glassimage/src/gls_cl.cpp
        glassimage/src/gls_ocl.cpp
        glassimage/src/gls_android_support.cpp
        glassimage/src/gls_image_png.cpp
        glassimage/src/gls_image_jpeg.cpp
        glassimage/src/gls_image_tiff.cpp
        glassimage/src/gls_tiff_metadata.cpp
        glassimage/src/gls_dng_lossless_jpeg.cpp
        glassimage/glsTestV2/glassImageTestV2.cpp)

target_compile_options( glassImageTest PRIVATE -Wall -Werror ) # -DUSE_ASSET_MANAGER

# Add header files for opencl
target_include_directories( glassImageTest PRIVATE ${CMAKE_SOURCE_DIR}/headers )
target_include_directories( glassImageTest PRIVATE ${CMAKE_SOURCE_DIR}/glassimage/include )

target_compile_options( glassImageTest PRIVATE -Wall -Werror -DUSE_IOSTREAM_LOG )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        glassImageTest

        libjpg
        libpng
        libz
        libtiff
        libtiffxx

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        -ljnigraphics
        -landroid)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        native-lib.cpp)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        log)