#include "glass_image/gpu_image_3d.h"

#include <format>
#include <stdexcept>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
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
      row_pitch_(gu::GetBestRowPitch<T>(width)),
      flags_(flags),
      buffer_(GpuBuffer<T>(gpu_context, row_pitch_ * height_ * width_, flags))
{
    image_ = CreateImage3dFromBuffer(buffer_, 0, row_pitch_, width_, height_, depth_, flags_);
}

template <typename T>
GpuImage3d<T>::GpuImage3d(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage3d<T>& other, std::optional<size_t> x0,
                          std::optional<size_t> y0, std::optional<size_t> z0, std::optional<size_t> width,
                          std::optional<size_t> height, std::optional<size_t> depth)
    : gpu_context_(gpu_context),
      width_(width.value_or(other.width_)),
      height_(height.value_or(other.height_)),
      depth_(depth.value_or(other.depth_)),
      row_pitch_(other.row_pitch_),  // Have to keep other's row pitch
      flags_(other.flags_),
      buffer_(other.buffer_)
{
    const size_t _x0 = x0.value_or(0);
    const size_t _y0 = y0.value_or(0);
    const size_t _z0 = z0.value_or(0);
    const size_t _width = width.value_or(other.width_);
    const size_t _height = height.value_or(other.height_);
    const size_t _depth = depth.value_or(other.depth_);

    if (_x0 + _width > other.width_ || _y0 + _height > other.height_ || _z0 + _height > other.height_)
        throw std::runtime_error(
            std::format("Image crop of [{}, {}, {}, {}, {}, {}] is out of bounds for source image shaped {}x{}x{}.",
                        _x0, _y0, _z0, _width, _height, _depth, other.width_, other.height_, other.depth_));

    /// TODO: This might potentially cause an error if your offset is wrong in exactly such a way as to cause an invalid
    /// buffer offset. Have to keep an eye on this.
    const size_t offset = _z0 * height_ * row_pitch_ + _y0 * row_pitch_ + _x0;  // In pixels
    image_ = CreateImage3dFromBuffer(buffer_, offset, row_pitch_, width_, height_, depth_, flags_);
}

// template <typename T>
// gls::image<T> GpuImage3d<T>::ToImage(const size_t z, std::optional<cl::CommandQueue> queue,
//                                      const std::vector<cl::Event>& events)
// {
//     cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
//     gls::image<T> host_image(width_, height_);
//     const size_t row_pitch = host_image.stride * sizeof(T);

//     _queue.enqueueReadImage(image_, CL_TRUE, {0, 0, z}, {width_, height_, 1}, row_pitch, 0,
//     host_image.pixels().data(),
//                             &events);
//     return host_image;
// }

// template <typename T>
// cl::Event GpuImage3d<T>::CopyFrom(const gls::image<T>& image, const size_t z, std::optional<cl::CommandQueue> queue,
//                                   const std::vector<cl::Event>& events)
// {
//     if (image.width != width_ || image.height != height_)
//         throw std::runtime_error(std::format("CopyFrom() expected image of size {}x{}, got {}x{}.", width_, height_,
//                                              image.width, image.height));

//     if (z >= depth_)
//         throw std::runtime_error(std::format("CopyFrom() z index {} is out of bounds for depth {}.", z, depth_));

//     cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
//     cl::Event event;
//     const size_t row_pitch = image.stride * sizeof(T);
//     _queue.enqueueWriteImage(image_, CL_FALSE, {0, 0, z}, {width_, height_, 1}, row_pitch, 0, image.pixels().data(),
//                              &events, &event);
//     return event;
// }

// template <typename T>
// cl::Event GpuImage3d<T>::CopyTo(gls::image<T>& image, const size_t z, std::optional<cl::CommandQueue> queue,
//                                 const std::vector<cl::Event>& events)
// {
//     if (image.width != width_ || image.height != height_)
//         throw std::runtime_error(std::format("CopyTo() expected image of size {}x{}, got {}x{}.", width_, height_,
//                                              image.width, image.height));

//     if (z >= depth_)
//         throw std::runtime_error(std::format("CopyTo() z index {} is out of bounds for depth {}.", z, depth_));

//     cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
//     cl::Event event;
//     const size_t row_pitch = image.stride * sizeof(T);
//     _queue.enqueueReadImage(image_, CL_FALSE, {0, 0, z}, {width_, height_, 1}, row_pitch, 0, image.pixels().data(),
//                             &events, &event);
//     return event;
// }

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
GpuImage<T> GpuImage3d<T>::operator[](const size_t z)
{
    if (z >= depth_)
        throw std::runtime_error(std::format("operator[] z index {} is out of bounds for depth {}.", z, depth_));

    // Calculate the offset for the slice at depth z
    const size_t offset = z * height_ * row_pitch_;  // In pixels

    return GpuImage<T>(gpu_context_, buffer_, width_, height_, offset);
}

template <typename T>
cl::Image3D GpuImage3d<T>::CreateImage3dFromBuffer(GpuBuffer<T>& buffer, const size_t offset, const size_t row_pitch,
                                                   const size_t width, const size_t height, const size_t depth,
                                                   cl_mem_flags flags)
{
    /* Based on @mako443's research, buffer based and non-buffer-based images have close to the same texture access
     * speeds on Adreno GPUs. If anything, I was able to find 10-20% slow downs in some edge cases. Therefore, we always
     * create images based on buffers to provide ahead-of-time buffer allocation and cropping functionality. To my
     * understanding, images have to adhere to the two device-required alignment constraints checked below.
     */

    cl::Device device = cl::Device::getDefault();
    cl_uint image_pitch_alignment = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();  // In pixels
    // cl_uint image_base_alignment = device.getInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>();  // In pixels, what for?
    cl_uint mem_base_alignment = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();  // In bits!

    const size_t offset_bytes = offset * sizeof(T);
    const size_t row_bytes = row_pitch * sizeof(T);
    const size_t mem_base_alignment_bytes = mem_base_alignment / 8;
    const size_t image_pitch_alignment_bytes = image_pitch_alignment * sizeof(T);

    if (offset_bytes % mem_base_alignment_bytes)
        throw std::runtime_error(std::format(
            "A buffer offset of {} pixels for type {} is invalid for device required base alignment of {} bytes.",
            offset, typeid(T).name(), mem_base_alignment_bytes));

    if (row_bytes % image_pitch_alignment_bytes)
        throw std::runtime_error(std::format(
            "A row pitch of {} pixels for type {} is invalid for device required pitch alignment of {} bytes",
            row_pitch, typeid(T).name(), image_pitch_alignment_bytes));

    if (row_pitch < width)
        throw std::runtime_error(
            std::format("Row pitch of {} pixels is smaller than width of {} pixels.", row_pitch, width));

    if (buffer.ByteSize() < offset_bytes + depth * height * row_bytes)
        throw std::runtime_error(
            std::format("Buffer size of {} bytes is too small for offset of {} bytes plus size of {} bytes.",
                        buffer.ByteSize(), offset_bytes, depth * height * row_bytes));

    // Create a sub buffer to set the initial image offset
    cl_int err;
    cl_buffer_region region{.origin = offset_bytes, .size = depth * height * row_bytes};
    cl::Buffer sub_buffer = buffer.buffer().createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err != CL_SUCCESS) throw std::runtime_error(std::format("Sub buffer creation failed with error code {}.", err));

    cl_image_desc image_desc;
    memset(&image_desc, 0, sizeof(image_desc));
    image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    image_desc.image_depth = depth;
    image_desc.buffer = sub_buffer();
    size_t pixel_size = sizeof(T);
    image_desc.image_row_pitch = row_bytes;
    image_desc.image_slice_pitch = height * row_bytes;  // Think no reason to set this differently for now.

    cl::ImageFormat format = gu::GetClFormat<T>();
    cl_image_format image_format;
    image_format.image_channel_order = format.image_channel_order;
    image_format.image_channel_data_type = format.image_channel_data_type;

#ifdef __APPLE__
    cl_mem image_mem =
        clCreateImage(gpu_context_->clContext().get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
#else
    cl_mem image_mem = opencl::clCreateImage(gpu_context_->clContext().get(), CL_MEM_READ_WRITE, &image_format,
                                             &image_desc, nullptr, &err);
#endif

    if (err != CL_SUCCESS)
        throw std::runtime_error(std::format("cl::Image2D creation from buffer failed with error code {}.", err));

    return cl::Image3D(image_mem);
}
// {
//     return cl::Image3D();
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
// }

template class GpuImage3d<float>;
template class GpuImage3d<gls::pixel_fp32>;
template class GpuImage3d<gls::pixel_fp32_2>;
template class GpuImage3d<gls::pixel_fp32_4>;
template class GpuImage3d<gls::luma_pixel_16>;

}  // namespace gls