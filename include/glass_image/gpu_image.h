#include <optional>
#include <span>
#include <string>
#include <vector>

#include "gls_image.hpp"
#include "gls_ocl.hpp"

namespace gls
{
template <typename T>
class GpuImage
{
   public:
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const std::array<size_t, 2> shape,
             cl_mem_flags flags = CL_MEM_READ_WRITE);
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const gls::image<T>& image,
             cl_mem_flags flags = CL_MEM_READ_WRITE);

    gls::image<T> ToImage(std::optional<cl::CommandQueue> queue = std::nullopt,
                          const std::vector<cl::Event>& events = {});

    cl::Event CopyFrom(const gls::image<T>& image, std::optional<cl::CommandQueue> queue = std::nullopt,
                       const std::vector<cl::Event>& events = {});

    const std::array<size_t, 2> shape_;

   private:
    cl::ImageFormat GetClFormat();

    std::shared_ptr<gls::OCLContext> gpu_context_;
    cl::Image2D image_;
    bool is_mapped_ = false;  // TODO: map image with the composed struct like buffer
};
}  // namespace gls