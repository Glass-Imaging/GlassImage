#include <OpenCL/cl.h>

#include <iostream>
#include <string>

#include "glass_image/gpu_buffer.h"
#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"

using namespace std;

int main()
{
    // gls::image<gls::rgb_pixel> image(512, 512);
    // gls::logging::current_log_level = gls::logging::LOG_LEVEL_INFO;
    // gls::logging::LogInfo("GlassImageTest") << "Image created: " << image.width << "x" << image.height << std::endl;

    auto gpu_context = make_shared<gls::OCLContext>(vector<string>{}, "");

    gls::image<float> some_image(8, 4);
    int value = 0;
    some_image.apply([&value](float* pixel, int x, int y) { *pixel = value++; });

    gls::GpuBuffer<float> buffer(gpu_context, CL_MEM_READ_WRITE, 4);
    cout << "Size: " << buffer.ToVector().size() << endl;

    gls::pixel_fp32_4 pix;
}