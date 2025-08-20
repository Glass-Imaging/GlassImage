#include "glass_image/gpu_image_3d.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

using std::vector;

// TEST(GpuImage3dTest, CopyFrom_CopyTo_ToImage)
// {
//     auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

//     gls::image<float> input_image(16, 4);

//     const size_t width = 16, height = 4, depth = 3;
//     gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);
//     EXPECT_EQ(gpu_image.width_, width);
//     EXPECT_EQ(gpu_image.height_, height);
//     EXPECT_EQ(gpu_image.depth_, depth);

//     // First slice is y * x
//     for (int y = 0; y < input_image.height; y++)
//         for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;
//     gpu_image.CopyFrom(input_image, 0).wait();

//     // Second slice is y + x
//     for (int y = 0; y < input_image.height; y++)
//         for (int x = 0; x < input_image.width; x++) input_image[y][x] = y + x;
//     gpu_image.CopyFrom(input_image, 1).wait();

//     // Third slice is x - y
//     for (int y = 0; y < input_image.height; y++)
//         for (int x = 0; x < input_image.width; x++) input_image[y][x] = x - y;
//     gpu_image.CopyFrom(input_image, 2).wait();

//     {
//         gls::image<float> cpu_image = gpu_image.ToImage(0);
//         cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, y * x); });
//     }

//     {
//         gls::image<float> cpu_image = gpu_image.ToImage(1);
//         cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, y + x); });
//     }

//     gls::image<float> cpu_image = gpu_image.ToImage(0);
//     gpu_image.CopyTo(cpu_image, 2).wait();
//     cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x - y); });
// }

TEST(GpuImage3dTest, SliceImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> cpu_image(16, 4);

    const size_t width = 16, height = 4, depth = 3;
    gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);

    for (int i = 0; i < depth; i++)
    {
        gls::GpuImage<float> slice = gpu_image[i];
        cpu_image.apply([&](float* pixel, int x, int y) { *pixel = x + i * y; });
        slice.CopyFrom(cpu_image).wait();
    }

    for (int i = 0; i < depth; i++)
    {
        gls::GpuImage<float> slice = gpu_image[i];
        gls::image<float> cpu_image = slice.ToImage();
        cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x + i * y); });
    }
}

TEST(GpuImage3dTest, CropOtherImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t width = 16, height = 4, depth = 3;
    gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);

    gls::GpuImage3d<float> gpu_crop(gpu_context, gpu_image, 0, 0, 0, 9, 3, 2);
}