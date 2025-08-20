#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "gls_image.hpp"
#include "gls_ocl.hpp"

namespace gls
{
template <typename T>
class GpuImage3d
{
   public:
    GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
               const size_t depth, cl_mem_flags flags = CL_MEM_READ_WRITE);

    GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage3d<T>& other, std::optional<size_t> x0 = {},
               std::optional<size_t> y0 = {}, std::optional<size_t> z0 = {}, std::optional<size_t> width = {},
               std::optional<size_t> height = {}, std::optional<size_t> depth = {});

    /* TODO: Would you guys like an API like my_image3d[z].CopyTo(my_gls_image) instead of
     * my_image3d.CopyTo(my_gls_image, z)? It would be doable, potentially even by slicing a gls::GpuImage on-the
     * fly! - that would be super cool.*/

    // gls::image<T> ToImage(const size_t z, std::optional<cl::CommandQueue> queue = std::nullopt,
    //                       const std::vector<cl::Event>& events = {});

    // cl::Event CopyFrom(const gls::image<T>& image, const size_t z, std::optional<cl::CommandQueue> queue =
    // std::nullopt,
    //                    const std::vector<cl::Event>& events = {});

    // cl::Event CopyTo(gls::image<T>& image, const size_t z, std::optional<cl::CommandQueue> queue = std::nullopt,
    //                  const std::vector<cl::Event>& events = {});

    cl::Event Fill(const T& value, std::optional<cl::CommandQueue> queue = std::nullopt,
                   const std::vector<cl::Event>& events = {});

    GpuImage<T> operator[](const size_t z);

    const size_t width_, height_, depth_, row_pitch_;
    cl::Image3D image() { return image_; };
    const cl::Image3D image() const { return image_; };

   private:
    // cl::Image3D CreateImage3dFromBuffer(GpuBuffer<T>& buffer, const size_t width, const size_t height,
    //                                     const size_t depth, cl_mem_flags flags,
    //                                     const std::optional<size_t> row_pitch_bytes = std::nullopt,
    //                                     const std::optional<size_t> slice_pitch_bytes = std::nullopt);

    cl::Image3D CreateImage3dFromBuffer(GpuBuffer<T>& buffer, const size_t offset, const size_t row_pitch,
                                        const size_t width, const size_t height, const size_t depth,
                                        cl_mem_flags flags);

    std::shared_ptr<gls::OCLContext> gpu_context_;

    GpuBuffer<T> buffer_;
    cl::Image3D image_;
    // bool is_mapped_ = false;  // TODO: map image with the composed struct like buffer
    const cl_mem_flags flags_;
};
}  // namespace gls