#pragma once

#include <optional>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "gls_ocl.hpp"

namespace gls
{
template <typename T>
class GpuImage3d
{
   public:
    GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
               const size_t depth, cl_mem_flags flags = CL_MEM_READ_WRITE);

    // Create from existing buffer
    GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, GpuBuffer<T>& buffer, const size_t width,
               const size_t height, const size_t depth, cl_mem_flags flags = CL_MEM_READ_WRITE);

// This constructor produces a segfault, is potentially hard to do reliably
#if false               
    GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage3d<T>& other, std::optional<size_t> x0 = {},
               std::optional<size_t> y0 = {}, std::optional<size_t> z0 = {}, std::optional<size_t> width = {},
               std::optional<size_t> height = {}, std::optional<size_t> depth = {});
#endif

    /* Implementing a slicing of GpuImage3d into GpuImage implicitly implements all gls::image interop methods without
     * specifying them here again. As gls::image has no 3D or array variant, more methods are probably not necessary at
     * this time and @mako443 has only left in Fil() for convenience. */

    GpuImage<T> operator[](const size_t z);

    cl::Event Fill(const T& value, std::optional<cl::CommandQueue> queue = std::nullopt,
                   const std::vector<cl::Event>& events = {});

    const size_t width_, height_, depth_, row_pitch_, slice_pitch_;
    cl::Image3D image() { return image_; };
    const cl::Image3D image() const { return image_; };
    const GpuBuffer<T> buffer() const { return buffer_; };

   private:
    cl::Image3D CreateImage3dFromBuffer(GpuBuffer<T>& buffer, const size_t offset, const size_t row_pitch,
                                        const size_t slice_pitch, const size_t width, const size_t height,
                                        const size_t depth, cl_mem_flags flags);

    std::shared_ptr<gls::OCLContext> gpu_context_;

    GpuBuffer<T> buffer_;
    cl::Image3D image_;
    const cl_mem_flags flags_;
    const std::optional<std::array<size_t, 3>> crop_region_ = std::nullopt;  // [x0, y0, z0] if this is already a crop.
};
}  // namespace gls