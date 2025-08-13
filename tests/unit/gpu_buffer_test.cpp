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