

#include <format>
#include <iostream>
#include <numeric>
#include <optional>
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
    cl::Buffer sub_buffer = buffer.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    assert(err == 0);

    cl::Image2D image(gpu_context->clContext(), format, sub_buffer, width, height, row_bytes, &err);
    assert(err == 0);

    return image;
}

int main()
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "", CL_QUEUE_PROFILING_ENABLE);
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");
    cl::Device device = cl::Device::getDefault();

    cl_uint pitchAlign = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();
    cl_uint baseAddrAlign = device.getInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>();

    const size_t width = 320, height = 160;
    const int dist = 21;

    cl::CommandQueue queue = gpu_context->clCommandQueue();

    {
        cl::Image2D image0(gpu_context->clContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width,
                           height);
        cl::Image2D image1(gpu_context->clContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width,
                           height);
        cl::Kernel kernel(gpu_context->clProgram(), "ReadIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("base", event);
    }

    {
        cl::Buffer buffer0(gpu_context->clContext(), CL_MEM_READ_WRITE, width * height * sizeof(float) * 4);
        cl::Buffer buffer1(gpu_context->clContext(), CL_MEM_READ_WRITE, width * height * sizeof(float) * 4);
        cl::Image2D image0 =
            CreateImage2dFromBuffer(gpu_context, buffer0, 0, width * sizeof(float) * 4, width, height,
                                    sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);
        cl::Image2D image1 =
            CreateImage2dFromBuffer(gpu_context, buffer1, 0, width * sizeof(float) * 4, width, height,
                                    sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);

        cl::Kernel kernel(gpu_context->clProgram(), "ReadIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("buffer-based", event);
    }

    {
        cl::Buffer buffer0(gpu_context->clContext(), CL_MEM_READ_WRITE,
                           (width + 4096) * (height + 256) * sizeof(float) * 4);
        cl::Buffer buffer1(gpu_context->clContext(), CL_MEM_READ_WRITE,
                           (width + 4096) * (height + 256) * sizeof(float) * 4);
        cl::Image2D image0 =
            CreateImage2dFromBuffer(gpu_context, buffer0, 128 * sizeof(float) * 4, width * sizeof(float) * 4, width,
                                    height, sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);
        cl::Image2D image1 =
            CreateImage2dFromBuffer(gpu_context, buffer1, 128 * sizeof(float) * 4, width * sizeof(float) * 4, width,
                                    height, sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);

        cl::Kernel kernel(gpu_context->clProgram(), "ReadIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("buffer-offset", event);
    }

    {
        cl::Buffer buffer0(gpu_context->clContext(), CL_MEM_READ_WRITE,
                           (width + 4096) * (height + 256) * sizeof(float) * 4);
        cl::Buffer buffer1(gpu_context->clContext(), CL_MEM_READ_WRITE,
                           (width + 4096) * (height + 256) * sizeof(float) * 4);
        cl::Image2D image0 =
            CreateImage2dFromBuffer(gpu_context, buffer0, 128 * sizeof(float) * 4, 4096 * sizeof(float) * 4, width,
                                    height, sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);
        cl::Image2D image1 =
            CreateImage2dFromBuffer(gpu_context, buffer1, 128 * sizeof(float) * 4, 4096 * sizeof(float) * 4, width,
                                    height, sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);

        cl::Kernel kernel(gpu_context->clProgram(), "ReadIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("buffer-offset-stride", event);
    }

    cl::Buffer buffer(gpu_context->clContext(), CL_MEM_READ_WRITE, width * height * sizeof(float) * 4);

    cout << endl << "All done." << endl;
}