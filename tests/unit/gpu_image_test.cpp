#include "glass_image/gpu_image.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

using std::vector, std::span;

TEST(GpuImageTest, CreateFromImage_ToImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    std::array<size_t, 2> shape{16, 4};
    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    gls::image<float> cpu_image = gpu_image.ToImage();         // Create CPU image out of GPU image

    EXPECT_EQ(gpu_image.shape_, shape);
    EXPECT_TRUE(cpu_image.width == shape[0] && cpu_image.height == shape[1]);

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, CreateFromImage_CopyTo)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    std::array<size_t, 2> shape{16, 4};
    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    gls::image<float> cpu_image(input_image.size());           // Create empty CPU image
    gpu_image.CopyTo(cpu_image).wait();                        // Copy the data

    EXPECT_EQ(gpu_image.shape_, shape);
    EXPECT_TRUE(cpu_image.width == shape[0] && cpu_image.height == shape[1]);

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, MapImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    std::array<size_t, 2> shape{16, 4};
    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    auto cpu_image = gpu_image.MapImage();                     // Map to CPU

    cpu_image->apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, ApplyOnCpu)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::image<float> input_image(16, 4);

    std::array<size_t, 2> shape{16, 4};
    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });     // Apply on CPU image
    gpu_image.ApplyOnCpu([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Apply on GPU image
    gls::image<float> cpu_image = gpu_image.ToImage();  // Create CPU image out of GPU image

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}
