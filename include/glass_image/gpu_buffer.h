#pragma once

#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "gls_ocl.hpp"

namespace gls
{

/// We implement GpuBuffer as a header-only library in order to allow for flexible, potentially struct-based buffer
/// types.
template <typename T>
class GpuBuffer
{
   public:
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, const size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE)
        : gpu_context_(gpu_context), size(size)
    {
        /// TODO: pitch for image? Or create croppable "buffer-based" images?
        buffer_ = cl::Buffer(gpu_context->clContext(), flags, sizeof(T) * size);
    };

    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, const std::span<T>& data,
              cl_mem_flags flags = CL_MEM_READ_WRITE)
        : gpu_context_(gpu_context), size(data.size())
    {
        buffer_ = cl::Buffer(gpu_context->clContext(), flags | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, data.data());
    };

    /// Warp the cl::Buffer
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl::Buffer buffer)
        : gpu_context_(gpu_context), size(buffer.getInfo<CL_MEM_SIZE>() / sizeof(T))
    {
        /// NOTE: Doing this with a buffer of custom types is potentially risky! I am not sure if the sizes align on
        /// both host and device sides - it does work in the unit test though.

        const size_t buffer_size = buffer.getInfo<CL_MEM_SIZE>();
        if (buffer_size % sizeof(T) != 0)
            throw std::runtime_error(
                std::format("Buffer of {} bytes does not evenly divide by type of size T.", buffer_size, sizeof(T)));

        buffer_ = buffer;  // Copies a reference to the underlying OpenCL buffer object.
    };

    std::vector<T> ToVector(std::optional<cl::CommandQueue> queue = std::nullopt,
                            const std::vector<cl::Event>& events = {})
    {
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        std::vector<T> data(size);
        _queue.enqueueReadBuffer(buffer_, CL_TRUE, 0, size * sizeof(T), data.data(), &events);
        return data;
    };

    cl::Event CopyFrom(const std::span<T>& data, std::optional<cl::CommandQueue> queue = std::nullopt,
                       const std::vector<cl::Event>& events = {})
    {
        if (data.size() != size)
            throw std::runtime_error(std::format("LoadVector() expected data of size {}, got {}.", size, data.size()));

        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueWriteBuffer(buffer_, CL_FALSE, 0, size * sizeof(T), data.data(), &events, &event);
        return event;
    };

    cl::Event CopyTo(std::span<T>& data, std::optional<cl::CommandQueue> queue = std::nullopt,
                     const std::vector<cl::Event>& events = {})
    {
        if (data.size() != size)
            throw std::runtime_error(std::format("LoadVector() expected data of size {}, got {}.", size, data.size()));

        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueReadBuffer(buffer_, CL_FALSE, 0, size * sizeof(T), data.data(), &events, &event);
        return event;
    }

    std::unique_ptr<std::span<T>, std::function<void(std::span<T>*)>> MapBuffer(
        std::optional<cl::CommandQueue> queue = std::nullopt, const std::vector<cl::Event>& events = {})
    {
        if (is_mapped_) throw std::runtime_error("MapBuffer() called on a buffer that is already mapped.");

        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        void* ptr = _queue.enqueueMapBuffer(buffer_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size * sizeof(T));

        // Create custom deleter that unmaps the buffer
        auto deleter = [this, ptr, _queue, events](std::span<T>* span) mutable
        {
            _queue.enqueueUnmapMemObject(buffer_, ptr, &events);
            is_mapped_ = false;
            delete span;
        };

        // Create span from mapped pointer
        auto mapped_span = new std::span<T>(static_cast<T*>(ptr), size);

        is_mapped_ = true;
        return std::unique_ptr<std::span<T>, std::function<void(std::span<T>*)>>(mapped_span, deleter);
    };

    const size_t size;
    size_t ByteSize() { return size * sizeof(T); };
    cl::Buffer buffer() { return buffer_; };

   private:
    std::shared_ptr<gls::OCLContext> gpu_context_;
    bool is_mapped_ = false;

    cl::Buffer buffer_;
};
}  // namespace gls