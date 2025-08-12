#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "glass_image/gpu_buffer.h"
#include "gls_image.hpp"
#include "gls_ocl.hpp"

namespace gls
{
template <typename T>
class GpuImage
{
   public:
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
             cl_mem_flags flags = CL_MEM_READ_WRITE);

    // Shallow copy constructor with same underlying OpenCL resources.
    GpuImage(const GpuImage& other);

    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const gls::image<T>& image,
             cl_mem_flags flags = CL_MEM_READ_WRITE);

    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& image, const size_t width, const size_t height);

    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& image, const size_t x0, const size_t y0,
             const size_t width, const size_t height);

    gls::image<T> ToImage(std::optional<cl::CommandQueue> queue = std::nullopt,
                          const std::vector<cl::Event>& events = {});

    cl::Event CopyFrom(const gls::image<T>& image, std::optional<cl::CommandQueue> queue = std::nullopt,
                       const std::vector<cl::Event>& events = {});

    cl::Event CopyTo(gls::image<T>& image, std::optional<cl::CommandQueue> queue = std::nullopt,
                     const std::vector<cl::Event>& events = {});

    cl::Event Fill(const T& value, std::optional<cl::CommandQueue> queue = std::nullopt,
                   const std::vector<cl::Event>& events = {});

    std::unique_ptr<gls::image<T>, std::function<void(gls::image<T>*)>> MapImage(
        std::optional<cl::CommandQueue> queue = std::nullopt, const std::vector<cl::Event>& events = {});

    void ApplyOnCpu(std::function<void(T* pixel, int x, int y)> process,
                    std::optional<cl::CommandQueue> queue = std::nullopt, const std::vector<cl::Event>& events = {});

    const size_t width_, height_;
    cl::Image2D image() { return image_; };

   private:
    cl::ImageFormat GetClFormat();
    std::tuple<size_t, size_t> GetPitches(const size_t width, const size_t height);
    size_t GetBufferSize(const size_t width, const size_t height);

    cl::Image2D CreateImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t width, const size_t height,
                                        cl_mem_flags flags);

    cl::Image2D CropImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t x0, const size_t y0, const size_t width,
                                      const size_t height, const size_t row_pitch_bytes, cl_mem_flags flags);

    std::shared_ptr<gls::OCLContext> gpu_context_;

    GpuBuffer<T> buffer_;
    cl::Image2D image_;
    bool is_mapped_ = false;  // TODO: map image with the composed struct like buffer
    const cl_mem_flags flags_;
};
}  // namespace gls