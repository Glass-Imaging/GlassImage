#include "glass_image/gpu_buffer.h"

#include <span>
#include <stdexcept>

#include "gls_image.hpp"

using std::vector, std::span, std::unique_ptr, std::format;

namespace gls
{
template <typename T>
GpuBuffer<T>::GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const size_t size)
    : gpu_context(gpu_context), size(size)
{
    /// TODO: pitch for image? Or create croppable "buffer-based" images?
    buffer = cl::Buffer(gpu_context->clContext(), flags, sizeof(T) * size);
};

template <typename T>
GpuBuffer<T>::GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const std::span<T>& data)
    : gpu_context(gpu_context), size(data.size())
{
    buffer = cl::Buffer(gpu_context->clContext(), flags | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, data.data());
};

template <typename T>
vector<T> GpuBuffer<T>::ToVector(std::optional<cl::CommandQueue> queue, const vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context->clCommandQueue());
    vector<T> data(size);
    _queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size * sizeof(T), data.data(), &events);
    return data;
};

template <typename T>
cl::Event GpuBuffer<T>::LoadVector(std::vector<T> data, std::optional<cl::CommandQueue> queue,
                                   const std::vector<cl::Event>& events)
{
    if (data.size() != size)
        throw std::runtime_error(format("LoadVector() expected data of size {}, got {}.", size, data.size()));

    cl::CommandQueue _queue = queue.value_or(gpu_context->clCommandQueue());
    cl::Event event;
    _queue.enqueueWriteBuffer(buffer, CL_FALSE, 0, size * sizeof(T), data.data(), &events, &event);
    return event;
}

template <typename T>
std::unique_ptr<MappedBuffer<T>> GpuBuffer<T>::MapBuffer(std::optional<cl::CommandQueue> queue,
                                                         const std::vector<cl::Event>& events)
{
    if (is_mapped) throw std::runtime_error("MapBuffer() called on a buffer that is already mapped.");

    cl::CommandQueue _queue = queue.value_or(gpu_context->clCommandQueue());
    void* ptr = _queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size * sizeof(T));
    auto cleanup = [this, ptr, &_queue, &events]()
    {
        _queue.enqueueUnmapMemObject(buffer, ptr, &events);
        is_mapped = false;
    };
    std::span<T> data_span(static_cast<T*>(ptr), size);

    is_mapped = true;
    return std::make_unique<MappedBuffer<T>>(data_span, cleanup);
}

template class GpuBuffer<float>;
template class GpuBuffer<gls::pixel_fp32>;
template class GpuBuffer<gls::pixel_fp32_2>;
template class GpuBuffer<gls::pixel_fp32_3>;
template class GpuBuffer<gls::pixel_fp32_4>;

}  // namespace gls