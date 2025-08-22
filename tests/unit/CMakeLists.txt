if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OPENCL_FRAMEWORK "-framework OpenCL")
else()
    set(OPENCL_FRAMEWORK "")
endif()

# gls::GpuBuffer test
add_executable(
  GpuBufferTest
  gpu_buffer_test.cpp
)

target_link_libraries(
    GpuBufferTest 
    GlassImage
    GTest::gtest_main
    ${OPENCL_FRAMEWORK}
)

# gls::GpuImage test
add_executable(
  GpuImageTest
  gpu_image_test.cpp
)

target_link_libraries(
    GpuImageTest 
    GlassImage
    GTest::gtest_main
    ${OPENCL_FRAMEWORK}
)

# gls::GpuImage3d test
add_executable(
  GpuImage3dTest
  gpu_image_3d_test.cpp
)

target_link_libraries(
    GpuImage3dTest 
    GlassImage
    GTest::gtest_main
    ${OPENCL_FRAMEWORK}
)

# gls::GpuKernel test
add_executable(
  GpuKernelTest
  gpu_kernel_test.cpp
)

target_link_libraries(
    GpuKernelTest 
    GlassImage
    GTest::gtest_main
    ${OPENCL_FRAMEWORK}
)

include(GoogleTest)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    gtest_discover_tests(GpuBufferTest)
    gtest_discover_tests(GpuImageTest)
    gtest_discover_tests(GpuImage3dTest)
    gtest_discover_tests(GpuKernelTest)
endif()