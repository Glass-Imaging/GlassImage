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
using gls::float16_t;

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

void PrintEvent(const std::string name, cl::Event event)
{
    event.wait();

    cl_ulong queue_time, start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queue_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
    cout << std::format("\t Trace {}: {} \n", name, (end_time - start_time) / 1000000.0);
}

cl::Image2D CreateImage2dFromBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl::Buffer buffer,
                                    const size_t offset_bytes, const size_t row_bytes, const size_t width,
                                    const size_t height, const size_t pixel_bytes, cl::ImageFormat format,
                                    cl_mem_flags flags)
{
    cl::Device device = cl::Device::getDefault();
    cl_uint image_pitch_alignment = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();  // In pixels
    // cl_uint image_base_alignment = device.getInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>();  // In pixels, what for?
    cl_uint mem_base_alignment = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();  // In bits!

    // std::cout << "image_pitch_alignment: " << image_pitch_alignment << std::endl;
    // std::cout << "mem_base_alignment: " << mem_base_alignment << std::endl;

    if (offset_bytes % (mem_base_alignment / 8)) throw std::runtime_error("Invalid offset bytes.");
    if (row_bytes % (image_pitch_alignment * pixel_bytes)) throw std::runtime_error("Invalid row bytes.");

    // Create sub-buffer from parent
    cl_int err;
    cl_buffer_region region{.origin = offset_bytes, .size = height * row_bytes};
    cl::Buffer sub_buffer = buffer.createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    assert(err == 0);

    cl::Image2D image(gpu_context->clContext(), format, sub_buffer, width, height, row_bytes, &err);
    assert(err == 0);

    return image;
}

void FillImage(shared_ptr<gls::OCLContext> gpu_context, cl::Image2D image, const size_t row_pitch = 0)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    cl::CommandQueue queue = gpu_context->clCommandQueue();
    size_t W, H, C;
    image.getImageInfo(CL_IMAGE_WIDTH, &W);
    image.getImageInfo(CL_IMAGE_HEIGHT, &H);

    const size_t data_w = row_pitch > 0 ? row_pitch / sizeof(float) / 4 : W;
    vector<float> data(data_w * H * 4);
    std::iota(data.begin(), data.end(), 0.0f);
    queue.enqueueWriteImage(image, CL_TRUE, {0, 0, 0}, {W, H, 1}, row_pitch, 0, data.data());
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cout << "Ela Fill: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl << endl;
}

int main()
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "", CL_QUEUE_PROFILING_ENABLE);
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");
    cl::Device device = cl::Device::getDefault();
    gls::image<float> cpu_image(16, 4);

    const size_t width = 16, height = 4, depth = 4;
    gls::GpuImage3d<float> gpu_image(gpu_context, width, height, depth);

    for (int i = 0; i < depth; i++)
    {
        gls::GpuImage<float> slice = gpu_image[i];
        cpu_image.apply([&](float* pixel, int x, int y) { *pixel = x + i * y; });
        slice.CopyFrom(cpu_image).wait();
    }

    for (int i = 0; i < depth; i++)
    {
        gls::GpuImage<float> slice = gpu_image[i];
        gls::image<float> cpu_image = slice.ToImage();
        // cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, x + i * y); });
    }

    cout << endl << "All done." << endl;
}