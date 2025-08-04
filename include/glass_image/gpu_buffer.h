#include <span>
#include <string>
#include <vector>

#include "gls_ocl.hpp"

namespace gls
{
template <typename T>
class GpuBuffer
{
   public:
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const size_t size);
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const std::span<T>& data);

    std::vector<T> ToVector();

   private:
    const size_t size;

    cl::Buffer buffer;
};
}  // namespace gls