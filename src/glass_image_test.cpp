
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "glass_image/gpu_kernel.h"
#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"
// #include "kernels.h"

using namespace std;

// class AddKernel : gls::GpuKernel
// {
//    public:
//     AddKernel(std::shared_ptr<gls::OCLContext> gpu_context) : gls::GpuKernel(gpu_context, "add_one") {}

//     cl::Event operator()(gls::GpuBuffer<float> buffer, int i, std::optional<cl::CommandQueue> queue = std::nullopt,
//                          const std::vector<cl::Event>& events = {})
//     {
//         assert(i == buffer.size);
//         SetArgs(buffer.buffer(), i);
//         cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
//         cl::Event event;
//         _queue.enqueueNDRangeKernel(kernel_, {}, {buffer.size, 1, 1}, {}, &events, &event);
//         return event;
//     }
// };

int main()
{
    cout << "HIIIIII" << endl;
    // auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    // gls::image<float> cpu_image(1024, 1024);
    // cpu_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Set values

    // gls::GpuImage<float> gpu_image(gpu_context, cpu_image);  // Create GPU image from CPU image
    // cout << gpu_image.width_ << "x" << gpu_image.height_ << endl;

    // gls::GpuImage<float> crop_image(gpu_context, gpu_image, 256, 256, 256, 256);

    // gls::image<float> out_image = crop_image.ToImage();
    // for (int y = 0; y < 4; y++)
    // {
    //     for (int x = 0; x < 4; x++)
    //     {
    //         cout << out_image[y][x] << ", ";
    //     }
    //     cout << endl;
    // }

    cout << endl << "All done." << endl;
}