
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"

using namespace std;

int main()
{
    auto gpu_context = std::make_shared<gls::OCLContext>(std::vector<std::string>{}, "");
    gls::GpuBuffer<float> buffer(gpu_context, CL_MEM_READ_WRITE, 4);

    auto mapped = buffer.MapBuffer();
    std::iota(mapped->data.begin(), mapped->data.end(), 0.0f);
    mapped.reset();

    std::vector<float> result = buffer.ToVector();

    // std::vector<float> result = buffer.ToVector();
}