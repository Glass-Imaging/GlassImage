#include <chrono>
#include <format>
#include <iostream>
#include <numeric>
#include <optional>
#include <ratio>
#include <stdexcept>
#include <string>
#include <vector>

#include "../tests/unit/testing_kernels.h"
#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "glass_image/gpu_image_3d.h"
#include "glass_image/gpu_kernel.h"
#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"

using namespace std;

int main()
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "", CL_QUEUE_PROFILING_ENABLE);
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");
    cl::Device device = cl::Device::getDefault();

    gls::image<float> input_image(16, 4);
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x * y); });  // Set values

    std::unique_ptr<gls::GpuImage<float>> gpu_image =
        std::make_unique<gls::GpuImage<float>>(gpu_context, input_image);  // Create GPU image from CPU image

    // Another, smaller GPU image with same buffer.
    gls::GpuImage<float> gpu_image2(gpu_context, *gpu_image, 0, 0, 7, 2);

    // gpu_image.reset();  // Delete the original image and check if the buffer is still intact after.

    // gls::image<float> cpu_image = gpu_image2.ToImage();  // Create CPU image out of GPU image
    // cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x * y); });
}

void other_check()
{
    // const size_t width = 1152, height = 8, depth = 3;
    // gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);

    // // Fill the image with increasing values via slices
    // float val = 0;
    // for (int z = 0; z < depth; z++)
    // {
    //     gls::GpuImage<float> gpu_slice = gpu_image[z];
    //     gls::image<float> cpu_slice = gpu_slice.ToImage();
    //     for (int y = 0; y < height; y++)
    //     {
    //         for (int x = 0; x < width; x++)
    //         {
    //             cpu_slice[y][x] = val++;
    //         }
    //     }
    //     gpu_slice.CopyFrom(cpu_slice).wait();
    // }

    // const size_t x0 = 0, y0 = 0, z0 = 1;
    // // gls::GpuImage3d<float> gpu_crop(gpu_context, gpu_image, x0, y0, z0, 6, 3, 2);

    // cout << "-------------" << endl;

    // vector<float> data = gpu_crop.buffer().ToVector();
    // cout << data[10240] << endl;
    // cout << data[10241] << endl;
    // cout << data[10242] << endl;

    // for (int z = 0; z < 1; z++)
    // {
    //     gls::GpuImage<float> gpu_slice = gpu_crop[z];
    //     gls::image<float> cpu_slice = gpu_slice.ToImage();

    //     for (int y = 0; y < 3; y++)
    //     {
    //         for (int x = 0; x < 6; x++)
    //         {
    //             const float expected = (z + z0) * width * height + (y + y0) * width + (x + x0);
    //             cout << cpu_slice[y][x] << ", ";
    //         }
    //         cout << endl;
    //     }
    // }

    // for (int i = 0; i < depth; i++)
    // {
    //     gls::GpuImage<float> slice = gpu_image[i];
    //     gls::image<float> cpu_image = slice.ToImage();
    //     cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x + i * y); });
    // }
}