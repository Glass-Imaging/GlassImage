# /bin/sh

# rm -fr build
mkdir -p build
cd build

cmake -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_USE_LIBC=1 ..

make glsImage
