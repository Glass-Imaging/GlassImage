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
    MappedBuffer(std::span<T> data, std::function<void()> cleanup) : data(data), cleanup(cleanup) {};
    ~MappedBuffer() { cleanup(); };

    std::span<T> data;

   private:
    std::function<void()> cleanup;
};

template <typename T>
class GpuBuffer
{
   public:
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const size_t size);
    GpuBuffer(std::shared_ptr<gls::OCLContext> gpu_context, cl_mem_flags flags, const std::span<T>& data);

    std::vector<T> ToVector(std::optional<cl::CommandQueue> queue = std::nullopt,
                            const std::vector<cl::Event>& events = {});

    cl::Event LoadVector(std::vector<T>, std::optional<cl::CommandQueue> queue = std::nullopt,
                         const std::vector<cl::Event>& events = {});

    std::unique_ptr<MappedBuffer<T>> MapBuffer(std::optional<cl::CommandQueue> queue = std::nullopt,
                                               const std::vector<cl::Event>& events = {});

   private:
    std::shared_ptr<gls::OCLContext> gpu_context;
    const size_t size;
    bool is_mapped = false;

    cl::Buffer buffer;
};
}  // namespace gls