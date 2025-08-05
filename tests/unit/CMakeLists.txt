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


include(GoogleTest)
gtest_discover_tests(GpuBufferTest)
gtest_discover_tests(GpuImageTest)