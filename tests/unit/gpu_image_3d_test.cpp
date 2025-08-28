#include "glass_image/gpu_image_3d.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using std::vector;

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

TEST(GpuImage3dTest, FromBuffer)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t width = 1280, height = 8, depth = 2;
    gls::GpuBuffer<float> buffer(gpu_context, width * height * depth * sizeof(float));
    vector<float> data(buffer.size_, 1.1f);
    buffer.CopyFrom(data);

    gls::GpuImage3d<float> gpu_image(gpu_context, buffer, width, height, depth);

    gls::GpuImage<float> slice = gpu_image[0];
    gls::image<float> cpu_image = slice.ToImage();
    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, 1.1f); });
}

// Currently proved unstable
#if false
TEST(GpuImage3dTest, CropOtherImage)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    const size_t width = 1024, height = 8, depth = 3;
    gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);

    // Fill the image with increasing values via slices
    float val = 0;
    for (int z = 0; z < depth; z++)
    {
        gls::GpuImage<float> gpu_slice = gpu_image[z];
        gls::image<float> cpu_slice = gpu_slice.ToImage();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cpu_slice[y][x] = val++;
            }
        }
        gpu_slice.CopyFrom(cpu_slice).wait();
    }

    const size_t x0 = 512, y0 = 0, z0 = 1;
    gls::GpuImage3d<float> gpu_crop(gpu_context, gpu_image, x0, y0, z0, 16, 6, 2);
    for (int z = 0; z < gpu_crop.depth_; z++)
    {
        gls::GpuImage<float> gpu_slice = gpu_image[z];
        gls::image<float> cpu_slice = gpu_slice.ToImage();

        for (int y = 0; y < gpu_crop.height_; y++)
        {
            for (int x = 0; x < gpu_crop.width_; x++)
            {
                const float expected = (z + z0) * width * height + (y + y0) * width + (x + x0);
                EXPECT_EQ(cpu_slice[y][x], expected);
            }
        }
    }
}
#endif