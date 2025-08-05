#include "glass_image/gpu_buffer.h"

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

using std::vector, std::span;

TEST(GpuBufferTest, CreateFromSpan_ToVector)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::GpuBuffer<float> buffer(gpu_context, CL_MEM_READ_WRITE, span<float>(data.data(), data.size()));

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), data.size());
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(data[i], result[i]);
}

TEST(GpuBufferTest, LoadVector)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");

    vector<float> data(6);
    std::iota(data.begin(), data.end(), 0.0f);

    gls::GpuBuffer<float> buffer(gpu_context, CL_MEM_READ_WRITE, data.size());
    buffer.LoadVector(data).wait();

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), data.size());
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(data[i], result[i]);
}

TEST(GpuBufferTest, MapBuffer)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");
    gls::GpuBuffer<float> buffer(gpu_context, CL_MEM_READ_WRITE, 4);

    auto mapped = buffer.MapBuffer();
    std::iota(mapped->data.begin(), mapped->data.end(), 0.0f);
    mapped.reset();

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), 4);
    for (int i = 0; i < result.size(); i++) EXPECT_EQ(i, result[i]);
}