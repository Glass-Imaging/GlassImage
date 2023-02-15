# /bin/sh

rm -fr build
mkdir build
cd build

cmake -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -std=c++20 -O3 ..

make glsImage

make glsTest
./glsTest ../../glsTest/Assets/baboon.tiff
