#include "glass_image/gpu_image_3d.h"

#include <CL/opencl.hpp>
#include <format>
#include <stdexcept>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_utils.h"
#include "gls_image.hpp"

namespace gu = gls::image_utils;

using std::cout, std::endl;

namespace gls
{

template <typename T>
GpuImage3d<T>::GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
                          const size_t depth, cl_mem_flags flags)
    : gpu_context_(gpu_context),
      width_(width),
      height_(height),
      depth_(depth),
      flags_(flags),
      buffer_(GpuBuffer<T>(gpu_context, gu::GetBestRowPitch<T>(width), flags))
{
    image_ = CreateImage3dFromBuffer(buffer_, width, height, depth, flags);
}

template <typename T>
GpuImage3d<T>::GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, const GpuImage3d<T>& other, const size_t width,
                          const size_t height, const size_t depth)
    : gpu_context_(gpu_context),
      width_(width),
      height_(height),
      depth_(depth),
      flags_(other.flags_),
      buffer_(other.buffer_)
{
    if (width > other.width_ || height > other.height_ || depth > other.depth_)
        throw std::logic_error(std::format("Cannot crop an image of size {}x{}x{} from source image of size {}x{}x{}.",
                                           width, height, depth, other.width_, other.height_, other.depth_));

    /// TODO: I think all these crop-copies need to carry a stride around! Otherwise you cannot do this in a chain.
    /// Implement and test this!
    // auto [row_pitch, slice_pitch] = gu::GetPitches<T>(other.width_, other.height_);
    // image_ = CreateImage3dFromBuffer(buffer_, width, height, depth, flags_, row_pitch, slice_pitch);
}

template <typename T>
gls::image<T> GpuImage3d<T>::ToImage(const size_t z, std::optional<cl::CommandQueue> queue,
                                     const std::vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    gls::image<T> host_image(width_, height_);
    const size_t row_pitch = host_image.stride * sizeof(T);

    _queue.enqueueReadImage(image_, CL_TRUE, {0, 0, z}, {width_, height_, 1}, row_pitch, 0, host_image.pixels().data(),
                            &events);
    return host_image;
}

template <typename T>
cl::Event GpuImage3d<T>::CopyFrom(const gls::image<T>& image, const size_t z, std::optional<cl::CommandQueue> queue,
                                  const std::vector<cl::Event>& events)
{
    if (image.width != width_ || image.height != height_)
        throw std::runtime_error(std::format("CopyFrom() expected image of size {}x{}, got {}x{}.", width_, height_,
                                             image.width, image.height));

    if (z >= depth_)
        throw std::runtime_error(std::format("CopyFrom() z index {} is out of bounds for depth {}.", z, depth_));

    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;
    const size_t row_pitch = image.stride * sizeof(T);
    _queue.enqueueWriteImage(image_, CL_FALSE, {0, 0, z}, {width_, height_, 1}, row_pitch, 0, image.pixels().data(),
                             &events, &event);
    return event;
}

template <typename T>
cl::Event GpuImage3d<T>::CopyTo(gls::image<T>& image, const size_t z, std::optional<cl::CommandQueue> queue,
                                const std::vector<cl::Event>& events)
{
    if (image.width != width_ || image.height != height_)
        throw std::runtime_error(std::format("CopyTo() expected image of size {}x{}, got {}x{}.", width_, height_,
                                             image.width, image.height));

    if (z >= depth_)
        throw std::runtime_error(std::format("CopyTo() z index {} is out of bounds for depth {}.", z, depth_));

    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;
    const size_t row_pitch = image.stride * sizeof(T);
    _queue.enqueueReadImage(image_, CL_FALSE, {0, 0, z}, {width_, height_, 1}, row_pitch, 0, image.pixels().data(),
                            &events, &event);
    return event;
}

template <typename T>
cl::Event GpuImage3d<T>::Fill(const T& value, std::optional<cl::CommandQueue> queue,
                              const std::vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;

    // TODO: Is there a more concise way?
    if constexpr (std::is_same_v<T, float>)
    {
        cl_float4 color = {value, 0.0f, 0.0f, 0.0f};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, depth_}, &events, &event);
    }
    else if constexpr (std::is_same_v<T, gls::pixel_fp32_2>)
    {
        cl_float4 color = {value[0], value[1], 0.0f, 0.0f};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, depth_}, &events, &event);
    }
    else if constexpr (std::is_same_v<T, gls::pixel_fp32_4>)
    {
        cl_float4 color = {value[0], value[1], value[2], value[3]};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, depth_}, &events, &event);
    }
    else if constexpr (std::is_same_v<T, gls::luma_pixel_16>)
    {
        cl_uint4 color = {value, 0, 0, 0};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, depth_}, &events, &event);
    }
    else
        throw std::runtime_error("Unsupported pixel type for GpuImage3d::Fill()");

    return event;
}

template <typename T>
cl::Image3D GpuImage3d<T>::CreateImage3dFromBuffer(GpuBuffer<T>& buffer, const size_t width, const size_t height,
                                                   const size_t depth, cl_mem_flags flags,
                                                   const std::optional<size_t> row_pitch_bytes,
                                                   const std::optional<size_t> slice_pitch_bytes)
{
    return cl::Image3D();
    //     /// NOTE: flags needs to match what the buffer was created with, but reading them from the buffer didn't work
    //     just
    //     /// now.
    //     auto [this_row_pitch, this_slice_pitch] = gu::GetPitches<T>(width, height);  // In bytes

    //     size_t row_pitch = row_pitch_bytes.value_or(this_row_pitch);
    //     size_t slice_pitch = slice_pitch_bytes.value_or(this_slice_pitch);

    //     const size_t expected_buffer_size = slice_pitch * depth;
    //     if (buffer.ByteSize() < expected_buffer_size)
    //         throw std::runtime_error(std::format("Expected a buffer of >= {} bytes as base for image, got {}.",
    //                                              expected_buffer_size, buffer.ByteSize()));

    //     cl_image_desc image_desc;
    //     memset(&image_desc, 0, sizeof(image_desc));
    //     image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    //     image_desc.image_width = width;
    //     image_desc.image_height = height;
    //     image_desc.image_depth = depth;
    //     image_desc.buffer = buffer.buffer().get();
    //     size_t pixel_size = sizeof(T);
    //     image_desc.image_row_pitch = row_pitch;
    //     image_desc.image_slice_pitch = slice_pitch;

    //     cl::ImageFormat format = gu::GetClFormat<T>();
    //     cl_image_format image_format;
    //     image_format.image_channel_order = format.image_channel_order;
    //     image_format.image_channel_data_type = format.image_channel_data_type;

    // #ifdef __APPLE__
    //     /*Creating an Image2D from a buffer fails on Mac, even with cl_khr_image2d_from_buffer explicitly listed.
    //     Therefore, I am returning a new cl::Image2D unrelated to the Buffer here. Note that this breaks having
    //     multiple images share the same buffer.
    //     */
    //     cl::Image3D image(gpu_context_->clContext(), flags, format, width, height, depth);
    //     return image;
    // #else
    //     cl_int err;
    //     cl_mem image_mem =
    //         opencl::clCreateImage(gpu_context_->clContext().get(), flags, &image_format, &image_desc, nullptr, &err);

    //     if (err != CL_SUCCESS)
    //     {
    //         std::stringstream ss;
    //         ss << "clCreateImage() failed in CreateImage3dFromBuffer()." << "  Error code: " << std::to_string(err)
    //            << "  Readable error code: " << gls::clStatusToString(err) << std::endl;
    //         throw cl::Error(err, ss.str().c_str());
    //     }

    //     // Wrap the cl_mem object in a cl::Image3D
    //     return cl::Image3D(image_mem);
    // #endif
}

template class GpuImage3d<float>;
template class GpuImage3d<gls::pixel_fp32>;
template class GpuImage3d<gls::pixel_fp32_2>;
template class GpuImage3d<gls::pixel_fp32_4>;
template class GpuImage3d<gls::luma_pixel_16>;

}  // namespace gls