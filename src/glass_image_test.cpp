
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
#include "kernels.h"

using namespace std;

class AddKernel : gls::GpuKernel
{
   public:
    AddKernel(std::shared_ptr<gls::OCLContext> gpu_context) : gls::GpuKernel(gpu_context, "add_one") {}

    cl::Event operator()(gls::GpuBuffer<float> buffer, int i, std::optional<cl::CommandQueue> queue = std::nullopt,
                         const std::vector<cl::Event>& events = {})
    {
        assert(i == buffer.size);
        SetArgs(buffer.buffer(), i);
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueNDRangeKernel(kernel_, {}, {buffer.size, 1, 1}, {}, &events, &event);
        return event;
    }
};

int main()
{
    std::vector<std::string> kernel_sources{kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "");
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "-DUSE_FLOAT16");

    gls::image<float> input_image(16, 4);
    input_image.apply([](float* pixel, int x, int y) { *pixel = static_cast<float>(x + y); });  // Set values
    gls::GpuBuffer<float> buffer(gpu_context, 16 * 4);

    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image

    // gls::GpuKernel kernel(gpu_context, "add_one");
    AddKernel kernel(gpu_context);

    // kernel.SetArgs(buffer.buffer(), 16 * 4);
    cl::Event event = kernel(buffer, 16 * 4);

    // gls::image<float> cpu_image = gpu_image2.ToImage();  // Create CPU image out of GPU image

    cout << endl << "All done." << endl;
}