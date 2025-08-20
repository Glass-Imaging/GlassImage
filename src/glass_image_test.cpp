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

    const size_t width = 16, height = 4, depth = 3;
    gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);

    int i = 2;
    {
        gls::GpuImage<float> slice = gpu_image[i];
        gls::image<float> cpu_image(16, 4);
        cpu_image.apply([&](float* pixel, int x, int y) { *pixel = x + i * y; });
        slice.CopyFrom(cpu_image).wait();
    }

    {
        gls::GpuImage<float> slice = gpu_image[i];
        gls::image<float> cpu_image = slice.ToImage();
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 6; x++)
            {
                cout << cpu_image[y][x] << ", ";
            }
            cout << endl;
        }
    }

    // for (int i = 0; i < depth; i++)
    // {
    //     gls::GpuImage<float> slice = gpu_image[i];
    //     gls::image<float> cpu_image = slice.ToImage();
    //     cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x + i * y); });
    // }

    cout << endl << "All done." << endl;
}