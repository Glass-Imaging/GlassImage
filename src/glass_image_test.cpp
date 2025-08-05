
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"

using namespace std;

int main()
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    std::array<size_t, 2> shape{16, 4};
    gls::GpuImage<float> gpu_image(gpu_context, input_image);

    auto cpu_image = gpu_image.MapImage();
}