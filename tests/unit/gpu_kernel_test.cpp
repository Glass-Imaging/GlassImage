
#include "glass_image/gpu_kernel.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "testing_kernels.h"

using std::vector;

class BufferAddKernel : gls::GpuKernel
{
   public:
    BufferAddKernel(std::shared_ptr<gls::OCLContext> gpu_context) : gls::GpuKernel(gpu_context, "BufferAddKernel") {}

    cl::Event operator()(gls::GpuBuffer<float> buffer, float value,
                         std::optional<cl::CommandQueue> queue = std::nullopt,
                         const std::vector<cl::Event>& events = {})
    {
        SetArgs(buffer.buffer(), value);
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueNDRangeKernel(kernel_, {}, {buffer.size, 1, 1}, {}, &events, &event);
        return event;
    }
};

class ImageAddKernel : gls::GpuKernel
{
   public:
    ImageAddKernel(std::shared_ptr<gls::OCLContext> gpu_context) : gls::GpuKernel(gpu_context, "ImageAddKernel") {}

    cl::Event operator()(gls::GpuImage<float> image, float value, std::optional<cl::CommandQueue> queue = std::nullopt,
                         const std::vector<cl::Event>& events = {})
    {
        SetArgs(image.image(), value, image.image());
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueNDRangeKernel(kernel_, {}, {image.width_, image.height_, 1}, {}, &events, &event);
        return event;
    }
};

TEST(GpuKernelTest, BufferKernel)
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "");
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");

    vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);
    gls::GpuBuffer<float> buffer(gpu_context, data);

    const float add_value = 1.5f;
    BufferAddKernel kernel(gpu_context);
    kernel(buffer, add_value).wait();

    vector<float> result = buffer.ToVector();
    for (int i = 0; i < data.size(); i++)
    {
        EXPECT_EQ(data[i] + add_value, result[i]);
    }
}

TEST(GpuKernelTest, ImageKernel)
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "");
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");

    gls::image<float> input_image(16, 4);
    for (int y = 0; y < input_image.height; y++)
        for (int x = 0; x < input_image.width; x++) input_image[y][x] = y * x;

    gls::GpuImage<float> gpu_image(gpu_context, input_image);  // Create GPU image from CPU image

    const float add_value = 1.5f;
    ImageAddKernel kernel(gpu_context);
    kernel(gpu_image, add_value).wait();

    gls::image<float> cpu_image = gpu_image.ToImage();

    cpu_image.apply([&](float* pixel, int x, int y) { EXPECT_EQ(*pixel, input_image[y][x] + add_value); });
}