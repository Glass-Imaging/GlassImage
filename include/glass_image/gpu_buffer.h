#include <optional>
#include <span>
#include <string>
#include <vector>

#include "gls_ocl.hpp"

namespace gls
{

template <typename T>
struct MappedBuffer final
{
   public:
    /// @brief Utility to keep track of a mapped buffer from GpuBuffer<T>::MapBuffer().
    /// @details Should only be used inside a unique_ptr to release mapped memory once it is not in use anymore.
    /// @param data Span covering the mapped host pointer.
    /// @param cleanup Cleanup function to call in destructor, use to unmap the buffer.
    MappedBuffer(std::span<T> data, std::function<void()> cleanup) : data_(data), cleanup_(cleanup) {};
    ~MappedBuffer() { cleanup_(); };

    std::span<T> data_;

   private:
    std::function<void()> cleanup_;
};

template <typename T>
class GpuBuffer
{
   public:
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, const size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, const std::span<T>& data,
              cl_mem_flags flags = CL_MEM_READ_WRITE);

    std::vector<T> ToVector(std::optional<cl::CommandQueue> queue = std::nullopt,
                            const std::vector<cl::Event>& events = {});

    cl::Event CopyFrom(const std::span<T>& data, std::optional<cl::CommandQueue> queue = std::nullopt,
                       const std::vector<cl::Event>& events = {});

    cl::Event CopyTo(std::span<T>& data, std::optional<cl::CommandQueue> queue = std::nullopt,
                     const std::vector<cl::Event>& events = {});

    /// TODO: Can't I use a custom deleter instead?!
    std::unique_ptr<MappedBuffer<T>> MapBuffer(std::optional<cl::CommandQueue> queue = std::nullopt,
                                               const std::vector<cl::Event>& events = {});

    const size_t size;

   private:
    std::shared_ptr<gls::OCLContext> gpu_context_;
    bool is_mapped_ = false;

    cl::Buffer buffer_;
};
}  // namespace gls