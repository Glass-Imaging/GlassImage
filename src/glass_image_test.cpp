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
    gls::GpuImage<float> gpu_image2(gpu_context, *gpu_image, 7, 3);  // Another, smaller GPU image with same buffer.

    gpu_image.reset();  // Delete the original image and check if the buffer is still intact after.

    gls::image<float> cpu_image = gpu_image2.ToImage();  // Create CPU image out of GPU image
    cpu_image.apply([&](float* pixel, int x, int y) { assert(*pixel == x * y); });

    cout << endl << "All done." << endl;
}