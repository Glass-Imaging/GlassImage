#include "glass_image/gpu_image.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

using std::vector;

TEST(GpuImageTest, CreateFromImage_ToImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    const size_t width = 16, height = 4;
    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    gls::image<float> cpu_image = gpu_image.ToImage();         // Create CPU image out of GPU image

    EXPECT_EQ(gpu_image.width_, width);
    EXPECT_EQ(gpu_image.height_, height);
    EXPECT_TRUE(cpu_image.width == width && cpu_image.height == height);

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, CreateFromImage_CopyTo)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    const size_t width = 16, height = 4;
    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    gls::image<float> cpu_image(input_image.size());           // Create empty CPU image
    gpu_image.CopyTo(cpu_image).wait();                        // Copy the data

    EXPECT_EQ(gpu_image.width_, width);
    EXPECT_EQ(gpu_image.height_, height);
    EXPECT_TRUE(cpu_image.width == width && cpu_image.height == height);

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, Fill)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t width = 16, height = 4;
    gls::GpuImage<float> gpu_image(gpu_context, width, height);  // Create GPU image from CPU image
    gpu_image.Fill(1.2f).wait();

    gls::image<float> cpu_image = gpu_image.ToImage();
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, 1.2f); });
}

TEST(GpuImageTest, MapImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    auto cpu_image = gpu_image.MapImage();                     // Map to CPU

    cpu_image->apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, ApplyOnCpu)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> input_image(16, 4);

    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });     // Apply on CPU image
    gpu_image.ApplyOnCpu([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Apply on GPU image
    gls::image<float> cpu_image = gpu_image.ToImage();  // Create CPU image out of GPU image

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
}

TEST(GpuImageTest, CropOtherImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> input_image(16, 4);
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Set values

    std::unique_ptr<gls::GpuImage<float>> gpu_image =
        std::make_unique<gls::GpuImage<float>>(gpu_context, input_image);  // Create GPU image from CPU image
    gls::GpuImage<float> gpu_image2(gpu_context, *gpu_image, 7, 3);  // Another, smaller GPU image with same buffer.

    gpu_image.reset();  // Delete the original image and check, if the buffer is still intact after.

    gls::image<float> cpu_image = gpu_image2.ToImage();  // Create CPU image out of GPU image

    /// Cropping an image from buffer does not work on Mac
#ifndef __APPLE__
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
#endif
}

#if false
// Still risky / incomplete!
TEST(GpuImageTest, CropOtherImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> cpu_image(1024, 1024);
    cpu_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Set values

    gls::GpuImage<float> gpu_image(gpu_context, cpu_image);  // Create GPU image from CPU image
    cout << gpu_image.width_ << "x" << gpu_image.height_ << endl;

    gls::GpuImage<float> crop_image(gpu_context, gpu_image, 256, 256, 256, 256);

    gls::image<float> out_image = crop_image.ToImage();
    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            cout << out_image[y][x] << ", ";
        }
        cout << endl;
    }

    /// Cropping an image from buffer does not work on Mac
#ifndef __APPLE__
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x]); });
#endif
}
#endif