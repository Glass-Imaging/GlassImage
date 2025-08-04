#include "glass_image/gpu_buffer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

TEST(GpuBufferTest, CreateBufferWithSize)
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");
    gls::GpuBuffer<float> buffer(gpu_context, CL_MEM_READ_WRITE, 4);

    std::vector<float> result = buffer.ToVector();
    EXPECT_EQ(result.size(), 4);
}