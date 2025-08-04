#include "glass_image/gpu_buffer.h"

#include <span>

#include "gls_image.hpp"

namespace gls
{
template <typename T>
GpuBuffer<T>::GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const size_t size)
    : size(size)
{
    /// TODO: pitch for image? Or create croppable "buffer-based" images?
    buffer = cl::Buffer(gpu_context->clContext(), flags, sizeof(T) * size);
};

template <typename T>
GpuBuffer<T>::GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const std::span<T>& data)
    : size(data.size())
{
    buffer = cl::Buffer(gpu_context->clContext(), flags | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, data.data());
};

template <typename T>
std::vector<T> GpuBuffer<T>::ToVector()
{
    return std::vector<T>(size);
};

template class GpuBuffer<float>;
template class GpuBuffer<gls::pixel_fp32>;
template class GpuBuffer<gls::pixel_fp32_2>;
template class GpuBuffer<gls::pixel_fp32_3>;
template class GpuBuffer<gls::pixel_fp32_4>;

}  // namespace gls