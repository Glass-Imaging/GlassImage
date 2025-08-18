#include "glass_image/gpu_buffer.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include "glass_image/gpu_kernel.h"
#include "testing_kernels.h"

using std::vector, std::span;

typedef struct
{
    int int_value;
    float float_value;
} CustomBufferStruct;

class CustomBufferAddKernel : gls::GpuKernel
{
   public:
    CustomBufferAddKernel(std::shared_ptr<gls::OCLContext> gpu_context)
        : gls::GpuKernel(gpu_context, "CustomBufferAddKernel")
    {
    }

    cl::Event operator()(gls::GpuBuffer<CustomBufferStruct> buffer,
                         std::optional<cl::CommandQueue> queue = std::nullopt,
                         const std::vector<cl::Event>& events = {})
    {
        SetArgs(buffer.buffer());
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueNDRangeKernel(kernel_, {}, {buffer.size, 1, 1}, {}, &events, &event);
        return event;
    }
};

TEST(GpuBufferTest, CreateFromSpan_ToVector)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::GpuBuffer<float> buffer(gpu_context, span<float>(data.data(), data.size()));

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), data.size());
    EXPECT_EQ(result.size(), buffer.size);
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(data[i], result[i]);
}

TEST(GpuBufferTest, CopyFrom)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::GpuBuffer<float> buffer(gpu_context, data.size());
    buffer.CopyFrom(data).wait();

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), data.size());
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(data[i], result[i]);
}

TEST(GpuBufferTest, CopyTo)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::GpuBuffer<float> buffer(gpu_context, data);
    vector<float> result(6);
    span<float> result_span(result.data(), result.size());
    buffer.CopyTo(result_span).wait();

    for (int i = 0; i < result.size(); i++) EXPECT_EQ(data[i], result[i]);
}

TEST(GpuBufferTest, MapBuffer)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");
    gls::GpuBuffer<float> buffer(gpu_context, 4);

    auto mapped = buffer.MapBuffer();
    std::iota(mapped->data_.begin(), mapped->data_.end(), 0.0f);
    mapped.reset();

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), 4);
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(i, result[i]);
}

TEST(GpuBufferTest, CustomBufferType)
{
    std::vector<std::string> kernel_sources{testing_kernel_code};
    auto gpu_context = std::make_shared<gls::OCLContext>(kernel_sources, "");
    gpu_context->loadProgramsFromFullStringSource(kernel_sources, "");

    gls::GpuBuffer<CustomBufferStruct> buffer(gpu_context, 4);

    auto mapped = buffer.MapBuffer();
    for (int i = 0; i < buffer.size; i++)
    {
        mapped->data_[i].int_value = i;
        mapped->data_[i].float_value = 0.2f;
    }
    mapped.reset();

    CustomBufferAddKernel kernel(gpu_context);
    kernel(buffer).wait();

    std::vector<CustomBufferStruct> result = buffer.ToVector();
    EXPECT_EQ(result.size(), 4);
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(result[i].float_value, i + 0.2f);
}