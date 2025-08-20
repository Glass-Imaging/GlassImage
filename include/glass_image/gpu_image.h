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
    // Keep default copy, move and destruction semantics, this class has only trivially copyable types.
    /// TODO: Cursor said this isn't ok here, but I don't follow the reasoning.
    GpuImage(const GpuImage&) = default;
    GpuImage& operator=(const GpuImage&) = default;
    GpuImage(GpuImage&&) = default;
    GpuImage& operator=(GpuImage&&) = default;
    ~GpuImage() = default;

    // New GpuImage
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
             cl_mem_flags flags = CL_MEM_READ_WRITE);

    // New GpuImage from gls::image
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const gls::image<T>& image,
             cl_mem_flags flags = CL_MEM_READ_WRITE);

    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuBuffer<T>& buffer, const size_t width,
             const size_t height, cl_mem_flags flags = CL_MEM_READ_WRITE);

    // Crop from another GpuImage, sharing same memory
    /// NOTE: The speed of this is usually fine but if you want to ensure a speed-optimal pitch, rather create from
    /// an existing GpuBuffer.
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& other, std::optional<size_t> x0 = std::nullopt,
             std::optional<size_t> y0 = std::nullopt, std::optional<size_t> width = std::nullopt,
             std::optional<size_t> height = std::nullopt);

#if false
    GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& image, const size_t x0, const size_t y0,
             const size_t width, const size_t height);
#endif

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
    const cl::Image2D image() const { return image_; };

   private:
    // cl::Image2D CreateImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t width, const size_t height,
    //                                     cl_mem_flags flags, const std::optional<size_t> row_pitch_bytes =
    //                                     std::nullopt, const std::optional<size_t> slice_pitch_bytes = std::nullopt);

    // cl::Image2D CropImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t x0, const size_t y0, const size_t width,
    //                                   const size_t height, const size_t row_pitch_bytes, cl_mem_flags flags);

    /// @brief Creates a 2D OpenCL image from a GPU buffer with specified offset and dimensions.
    /// @param buffer The GPU buffer to create the image from
    /// @param offset The starting offset in the buffer (in pixels).
    /// @param row_pitch The row pitch of the image (in pixels).
    /// @param width The width of the image (in pixels).
    /// @param height The height of the image (in pixels).
    /// @param flags OpenCL memory flags for the image.
    /// @return cl::Image2D object created from the buffer.
    cl::Image2D CreateImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t offset, const size_t row_pitch,
                                        const size_t width, const size_t height, cl_mem_flags flags);

    // General
    std::shared_ptr<std::atomic<bool>> is_mapped_;
    const cl_mem_flags flags_;
    const size_t row_pitch_;  // In pixels

    // GPU resources
    std::shared_ptr<gls::OCLContext> gpu_context_;
    GpuBuffer<T> buffer_;
    cl::Image2D image_;
};
}  // namespace gls