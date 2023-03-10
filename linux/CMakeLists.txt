cmake_minimum_required(VERSION 3.16) 

message(STATUS "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
if(CMAKE_USE_LIBC)
  message("*** Using libc++ runtime ***")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -stdlib=libc++")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message("*** Defaulting to Release build ***")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project("glsImage")

# CL will throw an unknown error on Linux if it is not explicitly linked with pthread, see flags at the bottom
find_package(Threads REQUIRED)

add_library( libz SHARED IMPORTED )
set_target_properties( libz PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libz.so )
add_library( libpng SHARED IMPORTED )
set_target_properties( libpng PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libpng.so )
add_library( libjpeg SHARED IMPORTED )
set_target_properties( libjpeg PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libjpeg.so ) #TODO: Possibly use jpeg-turbo

# Self-compiled libtiff
add_library( libtiff SHARED IMPORTED )
set_target_properties( libtiff PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libtiff.so )
add_library( libtiffxx SHARED IMPORTED )
set_target_properties( libtiffxx PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libtiffxx.so )


set(CMAKE_CXX_STANDARD 20)
set(ROOT_DIR ${CMAKE_SOURCE_DIR}/../)

###
# The following library and executable will probably not be used - actual library for bindings is compiled in other repo.
###

###
### Rendering library 
###

add_library(
    glsImage

    SHARED

    ${ROOT_DIR}/src/gls_cl.cpp
    ${ROOT_DIR}/src/gls_cl_error.cpp
    ${ROOT_DIR}/src/gls_color_science.cpp
    ${ROOT_DIR}/src/gls_dng_lossless_jpeg.cpp
    ${ROOT_DIR}/src/gls_image_jpeg.cpp
    ${ROOT_DIR}/src/gls_image_png.cpp
    ${ROOT_DIR}/src/gls_image_tiff.cpp
    ${ROOT_DIR}/src/gls_logging.cpp
    ${ROOT_DIR}/src/gls_tiff_metadata.cpp
)

target_include_directories( glsImage PRIVATE ${ROOT_DIR}/include ${ROOT_DIR}/glsTest )

target_link_libraries(
    glsImage
    libjpeg
    libpng
    libtiff
    libtiffxx
    libz
    OpenCL
)

###
### Demo pipeline
###

# Copy blur kernel
file(COPY ${ROOT_DIR}/glsTest/OpenCL/blur.cl DESTINATION ${ROOT_DIR}/linux/build/OpenCL)

add_executable(
    glsTest
    ${ROOT_DIR}/glsTest/cl_pipeline.cpp
    ${ROOT_DIR}/glsTest/glassImageTest.cpp
)

target_include_directories( glsTest PRIVATE ${ROOT_DIR}/include ${ROOT_DIR}/glsTest )

target_link_libraries(
    glsTest
    glsImage
)

# Setting pthread flags to prevent silent OpenCL error, works for g++ and Clang
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Werror=return-type")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
