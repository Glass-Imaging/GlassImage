#pragma once

#include <format>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "gls_ocl.hpp"

using std::cout, std::endl;

namespace gls
{

/// We implement GpuBuffer as a header-only library in order to allow for flexible, potentially struct-based buffer
/// types.
template <typename T>
class GpuBuffer
{
   public:
    // Keep default copy, move and destruction semantics, this class has only trivially copyable types.
    GpuBuffer(const GpuBuffer&) = default;
    GpuBuffer& operator=(const GpuBuffer&) = default;
    GpuBuffer(GpuBuffer&&) = default;
    GpuBuffer& operator=(GpuBuffer&&) = default;
    ~GpuBuffer() = default;

    // Create new
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, const size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE)
        : gpu_context_(gpu_context), size_(size), is_mapped_(std::make_shared<std::atomic<bool>>(false))
    {
        /// TODO: pitch for image? Or create croppable "buffer-based" images?
        buffer_ = cl::Buffer(gpu_context->clContext(), flags, sizeof(T) * size);
    };

    // Create from CPU span
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, const std::span<T>& data,
              cl_mem_flags flags = CL_MEM_READ_WRITE)
        : gpu_context_(gpu_context), size_(data.size()), is_mapped_(std::make_shared<std::atomic<bool>>(false))
    {
        buffer_ = cl::Buffer(gpu_context->clContext(), flags | CL_MEM_COPY_HOST_PTR, sizeof(T) * size_, data.data());
    };

    // Crop from another GpuBuffer
    /// NOTE: I don't know if this works recursively
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, GpuBuffer<T>& other, std::optional<size_t> offset = {},
              std::optional<size_t> size = {}, cl_mem_flags flags = CL_MEM_READ_WRITE)
        : gpu_context_(gpu_context),
          size_(size.value_or(other.size_ - offset.value_or(0))),
          is_mapped_(other.is_mapped_),
          is_crop_(true)
    {
        // Recursive cropping gave me a segfault - rather crop from the original again with appropriate offset.
        if (other.is_crop_) throw std::runtime_error("Cannot crop from a GpuBuffer that is already a crop.");

        const size_t _offset = offset.value_or(0);
        const size_t _size = size.value_or(other.size_ - _offset);

        if (_offset > other.size_ || _offset + _size > other.size_)
            throw std::runtime_error(
                std::format("Cropping with offset {} and size {} is invalid for source buffer of size {}.", _offset,
                            _size, other.size_));

        // Verify if the offset is legal
        cl::Device device = cl::Device::getDefault();
        cl_uint mem_base_alignment = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();  // In bits!
        const size_t offset_bytes = _offset * sizeof(T);
        const size_t mem_base_alignment_bytes = mem_base_alignment / 8;

        if (offset_bytes % mem_base_alignment_bytes)
            throw std::runtime_error(
                std::format("Cropping a buffer at {} pixels of type {} is invalid for device-required offset alignment "
                            "of {} bytes.",
                            _offset, typeid(T).name(), mem_base_alignment_bytes));

        cl_int err;
        cl_buffer_region region{.origin = _offset * sizeof(T), .size = (other.size_ - _offset) * sizeof(T)};
        buffer_ = other.buffer_.createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);

        if (err != CL_SUCCESS)
            throw std::runtime_error(std::format("Sub buffer creation failed with error code {}.", err));
    }

    /// Warp the cl::Buffer
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl::Buffer buffer)
        : gpu_context_(gpu_context),
          size_(buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
          is_mapped_(std::make_shared<std::atomic<bool>>(false)),
          buffer_(buffer)
    {
        /// NOTE: Doing this with a buffer of custom types is potentially risky! I am not sure if the sizes align on
        /// both host and device sides - it does work in the unit test though.

        const size_t buffer_size = buffer.getInfo<CL_MEM_SIZE>();
        if (buffer_size % sizeof(T) != 0)
            throw std::runtime_error(
                std::format("Buffer of {} bytes does not evenly divide by type of size T.", buffer_size, sizeof(T)));
    };

    std::vector<T> ToVector(std::optional<cl::CommandQueue> queue = std::nullopt,
                            const std::vector<cl::Event>& events = {}) const
    {
        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        std::vector<T> data(size_);
        _queue.enqueueReadBuffer(buffer_, CL_TRUE, 0, size_ * sizeof(T), data.data(), &events);
        return data;
    };

    cl::Event CopyFrom(const std::span<T>& data, std::optional<cl::CommandQueue> queue = std::nullopt,
                       const std::vector<cl::Event>& events = {})
    {
        if (data.size() != size_)
            throw std::runtime_error(std::format("LoadVector() expected data of size {}, got {}.", size_, data.size()));

        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueWriteBuffer(buffer_, CL_FALSE, 0, size_ * sizeof(T), data.data(), &events, &event);
        return event;
    };

    cl::Event CopyTo(std::span<T>& data, std::optional<cl::CommandQueue> queue = std::nullopt,
                     const std::vector<cl::Event>& events = {})
    {
        if (data.size() != size_)
            throw std::runtime_error(std::format("LoadVector() expected data of size {}, got {}.", size_, data.size()));

        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        cl::Event event;
        _queue.enqueueReadBuffer(buffer_, CL_FALSE, 0, size_ * sizeof(T), data.data(), &events, &event);
        return event;
    }

    std::unique_ptr<std::span<T>, std::function<void(std::span<T>*)>> MapBuffer(
        std::optional<cl::CommandQueue> queue = std::nullopt, const std::vector<cl::Event>& events = {})
    {
        if (is_mapped_->load()) throw std::runtime_error("MapBuffer() called on a buffer that is already mapped.");

        cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
        void* ptr = _queue.enqueueMapBuffer(buffer_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size_ * sizeof(T));

        // Create custom deleter that unmaps the buffer
        auto deleter = [this, ptr, _queue, events](std::span<T>* span) mutable
        {
            _queue.enqueueUnmapMemObject(buffer_, ptr, &events);
            is_mapped_->store(false);
            delete span;
        };

        // Create span from mapped pointer
        auto mapped_span = new std::span<T>(static_cast<T*>(ptr), size_);

        is_mapped_->store(true);
        return std::unique_ptr<std::span<T>, std::function<void(std::span<T>*)>>(mapped_span, deleter);
    };

    const size_t size_;
    size_t ByteSize() const { return size_ * sizeof(T); };
    cl::Buffer buffer() const { return buffer_; };
    const bool is_crop_ = false;

   private:
    std::shared_ptr<gls::OCLContext> gpu_context_;
    // Shared atomic bool such that multiple GpuBuffer, which wrap the same OpenCL memory, know if it is mapped or not.
    /// TODO: Potentially is_mapped_ would need to always consider the full buffer, I don't know if two images are
    /// allowed to be mapped at the same time if they share the same underlying buffer.
    std::shared_ptr<std::atomic<bool>> is_mapped_;

    cl::Buffer buffer_;
};
}  // namespace gls