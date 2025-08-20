#include "glass_image/gpu_image.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"

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

TEST(GpuImageTest, CopyConstructor)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t width = 16, height = 4;
    gls::GpuImage<float> gpu_image(gpu_context, width, height);  // Create GPU image from CPU image
    gpu_image.Fill(1.2f).wait();

    gls::GpuImage<float> other(gpu_image);

    gls::image<float> cpu_image = other.ToImage();
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
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x * y); });  // Set values

    std::unique_ptr<gls::GpuImage<float>> gpu_image =
        std::make_unique<gls::GpuImage<float>>(gpu_context, input_image);  // Create GPU image from CPU image

    // Another, smaller GPU image with same buffer.
    gls::GpuImage<float> gpu_image2(gpu_context, *gpu_image, 0, 0, 7, 2);

    gpu_image.reset();  // Delete the original image and check if the buffer is still intact after.

    gls::image<float> cpu_image = gpu_image2.ToImage();  // Create CPU image out of GPU image
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x * y); });
}

TEST(GpuImageTest, CropOtherImageOffset)
{
    // Same as above but crop with an offset
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> input_image(1024, 16);
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Set values

    std::unique_ptr<gls::GpuImage<float>> gpu_image =
        std::make_unique<gls::GpuImage<float>>(gpu_context, input_image);  // Create GPU image from CPU image
    gls::GpuImage<float> gpu_image2(gpu_context, *gpu_image, 512, 2, 16, 8);

    gpu_image.reset();  // Delete the original image and check if the buffer is still intact after.

    gls::image<float> cpu_image = gpu_image2.ToImage();  // Create CPU image out of GPU image
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, (x + 512) + (y + 2)); });
}

TEST(GpuImageTest, CreateFromBuffer)
{
    // Same as above but crop with an offset
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t w = 768, h = 4;  // Buffer must be large enough to contain pitch-adjusted image.
    vector<float> data(w * h);
    std::iota(data.begin(), data.end(), 0.0f);
    gls::GpuBuffer<float> buffer(gpu_context, data);

    gls::GpuImage<float> gpu_image(gpu_context, buffer, w, h);

    gls::image<float> cpu_image = gpu_image.ToImage();  // Create CPU image out of GPU image
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, y * w + x); });
}

TEST(GpuImageTest, PaddedPower2)
{
    // Same as above but crop with an offset
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t w = 512, h = 4;
    gls::GpuImage<float> gpu_image(gpu_context, w, h);

#if GLASS_IMAGE_PAD_POWER2_IMAGES
    EXPECT_EQ(gpu_image.row_pitch_, 768);
#else
    EXPECT_EQ(gpu_image.row_pitch_, 512);
#endif
}