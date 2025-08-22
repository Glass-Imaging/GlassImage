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

    cl_uint pitchAlign = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();
    cl_uint baseAddrAlign = device.getInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>();

    const size_t width = 4096, height = 3072;
    const int dist = 21;

    cl::CommandQueue queue = gpu_context->clCommandQueue();

    {
        cl::Image2D image0(gpu_context->clContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width,
                           height);
        cl::Image2D image1(gpu_context->clContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width,
                           height);
        cl::Kernel kernel(gpu_context->clProgram(), "WriteIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("base", event);

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        size_t row_pitch, slice_pitch;
        const size_t total_elements = width * height * 4;
        void* ptr = queue.enqueueMapImage(image0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width, height, 1},
                                          &row_pitch, &slice_pitch);
        std::span<float> data_span(static_cast<float*>(ptr), total_elements);
        std::iota(data_span.begin(), data_span.end(), 0.0f);
        queue.enqueueUnmapMemObject(image0, ptr, {}, &event);
        event.wait();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        cout << "Ela: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl << endl;

        FillImage(gpu_context, image0);
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

        cl::Kernel kernel(gpu_context->clProgram(), "WriteIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("buffer-based", event);

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        size_t row_pitch, slice_pitch;
        const size_t total_elements = width * height * 4;
        void* ptr = queue.enqueueMapImage(image0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width, height, 1},
                                          &row_pitch, &slice_pitch);
        std::span<float> data_span(static_cast<float*>(ptr), total_elements);
        std::iota(data_span.begin(), data_span.end(), 0.0f);
        queue.enqueueUnmapMemObject(image0, ptr, {}, &event);
        event.wait();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        cout << "Ela: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl << endl;

        FillImage(gpu_context, image0);
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

        cl::Kernel kernel(gpu_context->clProgram(), "WriteIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("buffer-offset", event);

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        size_t row_pitch, slice_pitch;
        const size_t total_elements = width * height * 4;
        void* ptr = queue.enqueueMapImage(image0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width, height, 1},
                                          &row_pitch, &slice_pitch);
        std::span<float> data_span(static_cast<float*>(ptr), total_elements);
        std::iota(data_span.begin(), data_span.end(), 0.0f);
        queue.enqueueUnmapMemObject(image0, ptr, {}, &event);
        event.wait();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        cout << "Ela: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl << endl;

        FillImage(gpu_context, image0);
    }

    {
        const size_t width_pixels = width + 4096;
        const size_t height_pixels = height + 128;
        // const size_t row_bytes = 4096 * sizeof(float) * 4;  // This is slow
        const size_t row_bytes = (width + 128) * sizeof(float) * 4;  // This is fast - presumably bc. *not* power
        // 2?!

        // const size_t width_pixels = width;
        // const size_t height_pixels = height;
        // const size_t row_bytes = width * sizeof(float) * 4;  // This is slow

        cl::Buffer buffer0(gpu_context->clContext(), CL_MEM_READ_WRITE,
                           width_pixels * height_pixels * sizeof(float) * 4);
        cl::Buffer buffer1(gpu_context->clContext(), CL_MEM_READ_WRITE,
                           width_pixels * height_pixels * sizeof(float) * 4);
        cl::Image2D image0 =
            CreateImage2dFromBuffer(gpu_context, buffer0, 0 * sizeof(float) * 4, row_bytes, width, height,
                                    sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);
        cl::Image2D image1 =
            CreateImage2dFromBuffer(gpu_context, buffer1, 0 * sizeof(float) * 4, row_bytes, width, height,
                                    sizeof(float) * 4, cl::ImageFormat(CL_RGBA, CL_FLOAT), CL_MEM_READ_WRITE);

        cl::Kernel kernel(gpu_context->clProgram(), "WriteIrregular2d");
        kernel.setArg(0, image0);
        kernel.setArg(1, dist);
        kernel.setArg(2, image1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, {width, height, 1}, {8, 8, 1}, {}, &event);
        PrintEvent("buffer-offset-stride", event);

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        size_t row_pitch, slice_pitch;
        const size_t total_elements = width * height * 4;
        void* ptr = queue.enqueueMapImage(image0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width, height, 1},
                                          &row_pitch, &slice_pitch);
        std::span<float> data_span(static_cast<float*>(ptr), total_elements);
        std::iota(data_span.begin(), data_span.end(), 0.0f);
        queue.enqueueUnmapMemObject(image0, ptr, {}, &event);
        event.wait();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        cout << "Ela: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl << endl;

        FillImage(gpu_context, image0, 0);
    }

    cout << endl << "All done." << endl;
}