#include "glass_image/gpu_kernel.h"

#include <stdexcept>

namespace gls
{

GpuKernel::GpuKernel(std::shared_ptr<gls::OCLContext> gpu_context, const std::string name)
    : gpu_context_(gpu_context), name_(name)
{
    try
    {
        kernel_ = cl::Kernel(gpu_context->clProgram(), name.c_str());
    }
    catch (const cl::Error& e)
    {
        throw std::runtime_error(format("Error loading kernel named {}.", name));
    }
}

}  // namespace gls