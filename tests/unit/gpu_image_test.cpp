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
