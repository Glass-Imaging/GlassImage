#include "glass_image/gpu_image.h"

#include <format>
#include <functional>
#include <stdexcept>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_utils.h"
#include "gls_image.hpp"

namespace gu = gls::image_utils;

using std::cout, std::endl;

namespace gls
{

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
                      cl_mem_flags flags)
    : gpu_context_(gpu_context),
      width_(width),
      height_(height),
      is_mapped_(std::make_shared<std::atomic<bool>>(false)),
      flags_(flags),
      row_pitch_(gu::GetBestRowPitch<T>(width)),
      buffer_(GpuBuffer<T>(gpu_context, row_pitch_ * height_, flags))
{
    image_ = CreateImage2dFromBuffer(buffer_, 0, row_pitch_, width_, height_, flags);
}

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const gls::image<T>& image, cl_mem_flags flags)
    : gpu_context_(gpu_context),
      width_(image.width),
      height_(image.height),
      is_mapped_(std::make_shared<std::atomic<bool>>(false)),
      flags_(flags),
      row_pitch_(gu::GetBestRowPitch<T>(image.width)),
      buffer_(GpuBuffer<T>(gpu_context, row_pitch_ * height_, flags))
{
    image_ = CreateImage2dFromBuffer(buffer_, 0, row_pitch_, width_, height_, flags_);
    CopyFrom(image).wait();
}

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuBuffer<T>& buffer, const size_t width,
                      const size_t height, cl_mem_flags flags)
    : gpu_context_(gpu_context),
      width_(width),
      height_(height),
      is_mapped_(std::make_shared<std::atomic<bool>>(false)),
      flags_(flags),
      row_pitch_(gu::GetBestRowPitch<T>(width)),
      buffer_(buffer)
{
    if (row_pitch_ * height > buffer.size_)
        throw std::runtime_error(
            std::format("GpuImage of size {}x{} with row pitch {} cannot be cropped from buffer of size {}.", width_,
                        height_, row_pitch_, buffer.size_));

    image_ = CreateImage2dFromBuffer(buffer, 0, row_pitch_, width_, height_, flags);
}

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& other, std::optional<size_t> x0,
                      std::optional<size_t> y0, std::optional<size_t> width, std::optional<size_t> height)
    : gpu_context_(gpu_context),
      width_(width.value_or(other.width_)),
      height_(height.value_or(other.height_)),
      is_mapped_(other.is_mapped_),
      flags_(other.flags_),
      row_pitch_(other.row_pitch_),  // Has to use the others row pitch
      buffer_(other.buffer_)
{
    const size_t _x0 = x0.value_or(0);
    const size_t _y0 = y0.value_or(0);
    const size_t _width = width.value_or(other.width_);
    const size_t _height = height.value_or(other.height_);

    if (_x0 + _width > other.width_ || _y0 + _height > other.height_)
        throw std::runtime_error(
            std::format("Image crop of [{}, {}, {}, {}] is out of bounds for source image shaped {}x{}.", _x0, _y0,
                        _width, _height, other.width_, other.height_));

    /// TODO: This might potentially cause an error if your offset is wrong in exactly such a way as to cause an invalid
    /// buffer offset. Have to keep an eye on this.
    const size_t offset = _y0 * row_pitch_ + _x0;  // In pixels
    image_ = CreateImage2dFromBuffer(buffer_, offset, row_pitch_, _width, _height, flags_);
}

#if false
// Cropping from a buffer region still seems finicky / risky due to alignment constraints
template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& image, const size_t x0,
                      const size_t y0, const size_t width, const size_t height)
    : gpu_context_(gpu_context), width_(width), height_(height), flags_(image.flags_), buffer_(image.buffer_)
{
    auto [row_pitch, slice_pitch] = GetPitches(image.width_, image.height_);
    image_ = CropImage2dFromBuffer(buffer_, x0, y0, width, height, row_pitch, image.flags_);
}
#endif

template <typename T>
gls::image<T> GpuImage<T>::ToImage(std::optional<cl::CommandQueue> queue, const std::vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    gls::image<T> host_image(width_, height_);
    const size_t row_pitch = host_image.stride * sizeof(T);

    _queue.enqueueReadImage(image_, CL_TRUE, {0, 0, 0}, {width_, height_, 1}, 0, 0, host_image.pixels().data(),
                            &events);
    return host_image;
}

template <typename T>
cl::Event GpuImage<T>::CopyFrom(const gls::image<T>& image, std::optional<cl::CommandQueue> queue,
                                const std::vector<cl::Event>& events)
{
    if (image.width != width_ || image.height != height_)
        throw std::runtime_error(std::format("LoadImage() expected image of size {}x{}, got {}x{}.", width_, height_,
                                             image.width, image.height));

    /// TODO: Potentially consider strides! Hopefully enough to set row pitch.
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;
    const size_t row_pitch = image.stride * sizeof(T);
    _queue.enqueueWriteImage(image_, CL_FALSE, {0, 0, 0}, {width_, height_, 1}, row_pitch, 0, image.pixels().data(),
                             &events, &event);  // NOTE: slice_pitch must be 0 for Image2D on Android.
    return event;
}

template <typename T>
cl::Event GpuImage<T>::CopyTo(gls::image<T>& image, std::optional<cl::CommandQueue> queue,
                              const std::vector<cl::Event>& events)
{
    if (image.width != width_ || image.height != height_)
        throw std::runtime_error(std::format("CopyTo() expected image of size {}x{}, got {}x{}.", width_, height_,
                                             image.width, image.height));

    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;
    const size_t row_pitch = image.stride * sizeof(T);
    _queue.enqueueReadImage(image_, CL_FALSE, {0, 0, 0}, {width_, height_, 1}, row_pitch, 0, image.pixels().data(),
                            &events, &event);  // NOTE: slice_pitch must be 0 for Image2D on Android.
    return event;
}

template <typename T>
std::unique_ptr<gls::image<T>, std::function<void(gls::image<T>*)>> GpuImage<T>::MapImage(
    std::optional<cl::CommandQueue> queue, const std::vector<cl::Event>& events)
{
    if (is_mapped_->load()) throw std::runtime_error("MapImage() called on an image that is already mapped.");

    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    size_t row_pitch, slice_pitch;
    void* ptr = _queue.enqueueMapImage(image_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width_, height_, 1},
                                       &row_pitch, &slice_pitch, &events);

    // Create custom deleter that unmaps the image
    auto deleter = [this, ptr, _queue, events](gls::image<T>* img) mutable
    {
        _queue.enqueueUnmapMemObject(image_, ptr, &events);
        is_mapped_->store(false);
        delete img;
    };

    // Create gls::image from mapped pointer
    const size_t stride = row_pitch / sizeof(T);
    const size_t total_elements = stride * height_;
    std::span<T> data_span(static_cast<T*>(ptr), total_elements);
    auto mapped_image = new gls::image<T>(width_, height_, stride, data_span);

    is_mapped_->store(true);
    return std::unique_ptr<gls::image<T>, std::function<void(gls::image<T>*)>>(mapped_image, deleter);
}

template <typename T>
void GpuImage<T>::ApplyOnCpu(std::function<void(T* pixel, int x, int y)> process, std::optional<cl::CommandQueue> queue,
                             const std::vector<cl::Event>& events)
{
    auto mapped_image = MapImage(queue, events);
    mapped_image->apply(process);
}

template <typename T>
cl::Event GpuImage<T>::Fill(const T& value, std::optional<cl::CommandQueue> queue, const std::vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;

    // TODO: Is there a more concise way?
    if constexpr (std::is_same_v<T, float>)
    {
        cl_float4 color = {value, 0.0f, 0.0f, 0.0f};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, 1}, &events, &event);
    }
    else if constexpr (std::is_same_v<T, pixel_fp32_2>)
    {
        cl_float4 color = {value[0], value[1], 0.0f, 0.0f};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, 1}, &events, &event);
    }
    else if constexpr (std::is_same_v<T, pixel_fp32_4>)
    {
        cl_float4 color = {value[0], value[1], value[2], value[3]};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, 1}, &events, &event);
    }
    else if constexpr (std::is_same_v<T, gls::luma_pixel_16>)
    {
        cl_uint4 color = {value, 0, 0, 0};
        _queue.enqueueFillImage(image_, color, {0, 0, 0}, {width_, height_, 1}, &events, &event);
    }
    else
        throw std::runtime_error("Unsupported pixel type for GpuImage::Fill()");

    return event;
}

// template <typename T>
// cl::ImageFormat GpuImage<T>::GetClFormat()
// {
//     if constexpr (std::is_same_v<T, float>)
//         return cl::ImageFormat(CL_R, CL_FLOAT);
//     else if constexpr (std::is_same_v<T, gls::pixel_fp32>)
//         return cl::ImageFormat(CL_R, CL_FLOAT);
//     else if constexpr (std::is_same_v<T, gls::pixel_fp32_2>)
//         return cl::ImageFormat(CL_RG, CL_FLOAT);
//     else if constexpr (std::is_same_v<T, gls::pixel_fp32_4>)
//         return cl::ImageFormat(CL_RGBA, CL_FLOAT);
//     else
//         throw std::runtime_error("Unsupported pixel type for GpuImage::GetClFormat()");
// }

// template <typename T>
// std::tuple<size_t, size_t> GpuImage<T>::GetPitches(const size_t width, const size_t height)
// {
//     /* TODO / NOTE: This is so far taken from GlassLibrary and seems to work.
//     However, @mako443 is not certain if 4096 is for sure the right alignment - is this related to the QCOM page size?
//     */

//     cl::Device device = cl::Device::getDefault();
//     auto image_pitch_alignment = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();
//     int row_pitch = width * sizeof(T);  // TODO: sizeof(T) is correct, right?

//     // Round up to the nearest multiple of image_pitch_alignment
//     row_pitch = (row_pitch + image_pitch_alignment - 1) & ~(image_pitch_alignment - 1);

//     // Round up to the nearest multiple of 4096
//     // Value 4096 is reverse engineered on how cl::Image objects are constructed by reading
//     // slice pitch values from cl::Image objects which owns their own memory.
//     // Did not find any correct documentation on this how it should be calculated on Qualcomm GPUs.
//     int slice_pitch = row_pitch * height;
//     slice_pitch = (slice_pitch + 4095) & ~4095;

//     return std::make_tuple(row_pitch, slice_pitch);
// }

// template <typename T>
// size_t GpuImage<T>::GetBufferSize(const size_t width, const size_t height, const size_t depth)
// {
//     auto [row_pitch, slice_pitch] = GetPitches(width, height);
//     return slice_pitch * depth;
// }

template <typename T>
cl::Image2D GpuImage<T>::CreateImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t offset, const size_t row_pitch,
                                                 const size_t width, const size_t height, cl_mem_flags flags)
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

    if (buffer.ByteSize() < offset_bytes + height * row_bytes)
        throw std::runtime_error(
            std::format("Buffer size of {} bytes is too small for offset of {} bytes plus size of {} bytes.",
                        buffer.ByteSize(), offset_bytes, height * row_bytes));

    // Create a sub buffer to set the initial image offset
    cl_int err;
    cl_buffer_region region{.origin = offset_bytes, .size = height * row_bytes};
    cl::Buffer sub_buffer = buffer.buffer().createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err != CL_SUCCESS) throw std::runtime_error(std::format("Sub buffer creation failed with error code {}.", err));

    // Create the cl::Image2D from the sub buffer
    cl::Image2D image(gpu_context_->clContext(), gu::GetClFormat<T>(), sub_buffer, width, height, row_bytes, &err);
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::format("cl::Image2D creation from buffer failed with error code {}.", err));

    return image;
}

// template <typename T>
// cl::Image2D GpuImage<T>::CreateImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t width, const size_t height,
//                                                  cl_mem_flags flags, const std::optional<size_t> row_pitch_bytes,
//                                                  const std::optional<size_t> slice_pitch_bytes)
// {
//     /// NOTE: flags needs to match what the buffer was created with, but reading them from the buffer didn't work
//     just
//     /// now.
//     auto [this_row_pitch, this_slice_pitch] = gu::GetPitches<T>(width, height);  // In bytes
//     size_t row_pitch = row_pitch_bytes.value_or(this_row_pitch);
//     size_t slice_pitch = slice_pitch_bytes.value_or(this_slice_pitch);

//     if (buffer.ByteSize() < slice_pitch)
//         throw std::runtime_error(
//             std::format("Expected a buffer of >= {} bytes as base for image, got {}.", slice_pitch,
//             buffer.ByteSize()));

//     cl_image_desc image_desc;
//     memset(&image_desc, 0, sizeof(image_desc));
//     image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
//     image_desc.image_width = width;
//     image_desc.image_height = height;
//     image_desc.buffer = buffer.buffer().get();
//     // size_t pixel_size = sizeof(T);
//     image_desc.image_row_pitch = row_pitch;
//     image_desc.image_slice_pitch = 0;  // TODO: This or slice_pitch? Image2D often wants slice pitch of 0, e.g. in
//     copy

//     cl::ImageFormat format = gu::GetClFormat<T>();
//     cl_image_format image_format;
//     image_format.image_channel_order = format.image_channel_order;
//     image_format.image_channel_data_type = format.image_channel_data_type;

// #ifdef __APPLE__
//     /*Creating an Image2D from a buffer fails on Mac, even with cl_khr_image2d_from_buffer explicitly listed.
//     Therefore, I am returning a new cl::Image2D unrelated to the Buffer here. Note that this breaks having multiple
//     images share the same buffer.
//     */
//     cl::Image2D image(gpu_context_->clContext(), flags, format, width, height);
//     return image;
// #else
//     cl_int err;
//     cl_mem image_mem =
//         opencl::clCreateImage(gpu_context_->clContext().get(), flags, &image_format, &image_desc, nullptr, &err);

//     if (err != CL_SUCCESS)
//     {
//         std::stringstream ss;
//         ss << "clCreateImage() failed in CreateImage2dFromBuffer()." << "  Error code: " << std::to_string(err)
//            << "  Readable error code: " << gls::clStatusToString(err) << std::endl;
//         throw cl::Error(err, ss.str().c_str());
//     }

//     // Wrap the cl_mem object in a cl::Image2D
//     return cl::Image2D(image_mem);
// #endif
// }

// template <typename T>
// cl::Image2D GpuImage<T>::CropImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t x0, const size_t y0,
//                                                const size_t width, const size_t height, const size_t row_pitch_bytes,
//                                                cl_mem_flags flags)
// {
//     // NOTE: flags needs to match what the buffer was created with, but reading them from the buffer didn't work just
//     // now.
//     const size_t buffer_start = y0 * row_pitch_bytes + x0 * sizeof(T);                           // In bytes
//     const size_t min_buffer_size = buffer_start + height * row_pitch_bytes + width / sizeof(T);  // In bytes
//     // std::cout << "SIZE " << buffer_start << ", " << height << ", " << row_pitch << ", " << width << std::endl;

//     if (buffer.size_ * sizeof(T) < min_buffer_size)
//         throw std::runtime_error(std::format(
//             "Expected buffer of at least {} bytes to crop at [{}, {}, {}, {}] with row pitch of {} bytes. Got {}.",
//             min_buffer_size, x0, y0, width, height, row_pitch_bytes, buffer.size_ * sizeof(T)));

//     cl::ImageFormat format = gu::GetClFormat<T>();

// #ifdef __APPLE__
//     /*Creating an Image2D from a buffer fails on Mac, even with cl_khr_image2d_from_buffer explicitly listed.
//     Therefore, I am returning a new cl::Image2D unrelated to the Buffer here. Note that this breaks having multiple
//     images share the same buffer.
//     */
//     cl::Image2D image(gpu_context_->clContext(), flags, format, width, height);
//     return image;
// #else
//     // Setting a buffer start offsets needs a sub buffer which only works through the C API.
//     cl_buffer_region region;
//     cl_int err;
//     region.origin = buffer_start * sizeof(T);   // In bytes
//     region.size = min_buffer_size * sizeof(T);  // In bytes
//     cl_mem sub_buffer =
//         opencl::clCreateSubBuffer(buffer.buffer().get(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
//     if (err != CL_SUCCESS) throw std::runtime_error(std::format("opencl::clCreateSubBuffer failed with code{}.",
//     err));

//     cl_image_desc image_desc;
//     memset(&image_desc, 0, sizeof(image_desc));
//     image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
//     image_desc.image_width = width;
//     image_desc.image_height = height;
//     image_desc.buffer = sub_buffer;
//     size_t pixel_size = sizeof(T);
//     image_desc.image_row_pitch = row_pitch_bytes;
//     image_desc.image_slice_pitch = 0;  // TODO: This or slice_pitch? Image2D often wants slice pitch of 0, e.g. in
//     copy

//     cl_image_format image_format;
//     image_format.image_channel_order = format.image_channel_order;
//     image_format.image_channel_data_type = format.image_channel_data_type;

//     cl_mem image_mem =
//         opencl::clCreateImage(gpu_context_->clContext().get(), flags, &image_format, &image_desc, nullptr, &err);

//     if (err != CL_SUCCESS)
//     {
//         std::stringstream ss;
//         ss << "clCreateImage() failed in CreateImage2dFromBuffer()." << "  Error code: " << std::to_string(err)
//            << "  Readable error code: " << gls::clStatusToString(err) << std::endl;
//         throw cl::Error(err, ss.str().c_str());
//     }

//     // Wrap the cl_mem object in a cl::Image2D
//     return cl::Image2D(image_mem);
// #endif
// }

template class GpuImage<float>;
// template class GpuImage<float16_t>;
// template class GpuImage<gls::pixel_fp16>;
// template class GpuImage<gls::pixel_fp16_2>;
// template class GpuImage<gls::pixel_fp16_3>; /// NOTE: We explicitly leave out RGB images because they show faulty
// behaviour on Android.
// template class GpuImage<gls::pixel_fp16_4>;
template class GpuImage<gls::pixel_fp32>;
template class GpuImage<gls::pixel_fp32_2>;
// template class GpuImage<gls::pixel_fp32_3>; /// NOTE: We explicitly leave out RGB images because they show faulty
// behaviour on Android.
template class GpuImage<gls::pixel_fp32_4>;
template class GpuImage<gls::luma_pixel_16>;

}  // namespace gls