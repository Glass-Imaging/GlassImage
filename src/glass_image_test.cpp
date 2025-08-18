
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#include <format>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "glass_image/gpu_image_3d.h"
#include "glass_image/gpu_kernel.h"
#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"
#include "../tests/unit/testing_kernels.h"

using namespace std;

class ReadIrregular2d : gls::GpuKernel
{
   public:
    ReadIrregular2d(std::shared_ptr<gls::OCLContext> gpu_context) : gls::GpuKernel(gpu_context, "ReadIrregular2d") {}

    cl::Event operator()(gls::GpuImage<gls::pixel_fp32_4> image, const int dist,
                         std::optional<cl::CommandQueue> queue = std::nullopt,
                         const std::vector<cl::Event>& events = {})
    {
        SetArgs(image.image(), dist);
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueNDRangeKernel(kernel_, {}, {image.width_, image.height_, 1}, {}, &events, &event);
        return event;
    }
};

void PrintEvent(cl::Event event)
{
    event.wait();

    cl_ulong queue_time, start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queue_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
    cout << std::format("\t Trace: {} \n", (start_time - queue_time) / 1000000.0);
}

int main()
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "", CL_QUEUE_PROFILING_ENABLE);
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");

    cl::CommandQueue queue = cl::CommandQueue::getDefault();
    cl::Image2D image0(gpu_context->clContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), 4096, 3072);
    cl::Image2D image1(gpu_context->clContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), 4096, 3072);
    cl::Kernel kernel(gpu_context->clProgram(), "ReadIrregular2d");
    kernel.setArg(0, image0);
    kernel.setArg(1, (int)9);
    kernel.setArg(2, image1);
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, {4096, 3072, 1}, {8, 8, 1}, {}, &event);

    cout << endl << "All done." << endl;
}