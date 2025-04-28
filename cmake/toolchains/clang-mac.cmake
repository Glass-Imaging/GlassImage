# System
set(CMAKE_SYSTEM_NAME MacOS)

# Compiler
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_FLAGS_INIT "-std=c++20 -stdlib=libc++ -O3")

# Options
set(GLASS_IMAGE_BUILD_IMAGE_IO ON) 