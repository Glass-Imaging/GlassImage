add_executable(
  GpuBufferTest
  gpu_buffer_test.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(
        GpuBufferTest 
        GlassImage
        GTest::gtest_main
        "-framework OpenCL" # TODO: Factor out to var
    )
else()
    target_link_libraries(
        GpuBufferTest 
        GlassImage
        GTest::gtest_main
    )
endif()

add_executable(
  GpuImageTest
  gpu_image_test.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(
        GpuImageTest 
        GlassImage
        GTest::gtest_main
        "-framework OpenCL" # TODO: Factor out to var
    )
else()
    target_link_libraries(
        GpuImageTest 
        GlassImage
        GTest::gtest_main
    )
endif()

add_executable(
  GpuKernelTest
  gpu_kernel_test.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(
        GpuKernelTest 
        GlassImage
        GTest::gtest_main
        "-framework OpenCL" # TODO: Factor out to var
    )
else()
    target_link_libraries(
        GpuKernelTest 
        GlassImage
        GTest::gtest_main
    )
endif()


include(GoogleTest)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    gtest_discover_tests(GpuBufferTest)
    gtest_discover_tests(GpuImageTest)
    gtest_discover_tests(GpuKernelTest)
endif()