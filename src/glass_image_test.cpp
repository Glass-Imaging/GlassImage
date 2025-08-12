
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "glass_image/gpu_image_3d.h"
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
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    gls::image<float> input_image(16, 4);

    const size_t width = 16, height = 4;
    gls::GpuImage3d<float> gpu_image(gpu_context, width, height, 2);

    // First slice is y * x
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;
    gpu_image.CopyFrom(input_image, 0);

    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y + x;
    gpu_image.CopyFrom(input_image, 1);

    gls::image<float> cpu_image = gpu_image.ToImage(1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            cout << cpu_image[y][x] << ", ";
        }
        cout << endl;
    }
    cpu_image.apply([&](float* pixel, int x, int y) { assert(*pixel == y + x); });

    // // First slice is y * x
    // for (int y = 0; y < input_image.height; y++)
    //     for (int x = 0; x < input_image.width; x++) input_image[y][x] = y + x;
    // gpu_image.CopyFrom(input_image, 1);

    cout << endl << "All done." << endl;
}