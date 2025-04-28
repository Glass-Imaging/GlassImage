project("glass_image")

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    cmake_minimum_required(VERSION 3.27)
    set(CMAKE_CXX_STANDARD 20)
    
    # TODO: Split Mac / Linux
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    message("Using compilers: ${CMAKE_C_COMPILER} / ${CMAKE_CXX_COMPILER}.")    
endif()

add_subdirectory(src)