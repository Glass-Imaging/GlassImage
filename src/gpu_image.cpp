#include "glass_image/gpu_image.h"

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
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const size_t width, const size_t height,
                      cl_mem_flags flags)
    : gpu_context_(gpu_context),
      width_(width),
      height_(height),
      flags_(flags),
      buffer_(GpuBuffer<T>(gpu_context, GetBufferSize(width, height), flags))
{
    // image_ = cl::Image2D(gpu_context->clContext(), flags, GetClFormat(), width, height);
    // std::cout << "IMAGE RAW" << std::endl;
    image_ = CreateImage2dFromBuffer(buffer_, width, height, flags);
}

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const gls::image<T>& image, cl_mem_flags flags)
    : gpu_context_(gpu_context),
      width_(image.width),
      height_(image.height),
      flags_(flags),
      buffer_(GpuBuffer<T>(gpu_context, GetBufferSize(image.width, image.height), flags))
{
    // image_ = cl::Image2D(gpu_context->clContext(), flags, GetClFormat(), image.width, image.height);
    image_ = CreateImage2dFromBuffer(buffer_, image.width, image.height, flags);
    CopyFrom(image).wait();
}

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, GpuImage<T>& image, const size_t width,
                      const size_t height)
    : gpu_context_(gpu_context), width_(width), height_(height), flags_(image.flags_), buffer_(image.buffer_)
{
    // image_ = cl::Image2D(gpu_context->clContext(), flags_, GetClFormat(), width_, height_);
    if (width > image.width_ || height > image.height_)
        throw std::logic_error(std::format("Cannot crop an image of size {}x{} from source image of size {}x{}.", width,
                                           height, image.width_, image.height_));
    image_ = CreateImage2dFromBuffer(buffer_, width, height, flags_);
}

template <typename T>
gls::image<T> GpuImage<T>::ToImage(std::optional<cl::CommandQueue> queue, const std::vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    gls::image<T> host_image(width_, height_);
    const size_t row_pitch = host_image.stride * sizeof(T);

    _queue.enqueueReadImage(image_, CL_TRUE, {0, 0, 0}, {width_, height_, 1}, row_pitch, 0, host_image.pixels().data(),
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
    if (is_mapped_) throw std::runtime_error("MapImage() called on an image that is already mapped.");

    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    size_t row_pitch, slice_pitch;
    void* ptr = _queue.enqueueMapImage(image_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0}, {width_, height_, 1},
                                       &row_pitch, &slice_pitch, &events);

    // Create custom deleter that unmaps the image
    auto deleter = [this, ptr, _queue, events](gls::image<T>* img) mutable
    {
        _queue.enqueueUnmapMemObject(image_, ptr, &events);
        is_mapped_ = false;
        delete img;
    };

    // Create gls::image from mapped pointer
    const size_t stride = row_pitch / sizeof(T);
    const size_t total_elements = stride * height_;
    std::span<T> data_span(static_cast<T*>(ptr), total_elements);
    auto mapped_image = new gls::image<T>(width_, height_, stride, data_span);

    is_mapped_ = true;
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
cl::ImageFormat GpuImage<T>::GetClFormat()
{
    if constexpr (std::is_same_v<T, float>)
        return cl::ImageFormat(CL_R, CL_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp32>)
        return cl::ImageFormat(CL_R, CL_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp32_2>)
        return cl::ImageFormat(CL_RG, CL_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp32_4>)
        return cl::ImageFormat(CL_RGBA, CL_FLOAT);
    else
        throw std::runtime_error("Unsupported pixel type for GpuImage::GetClFormat()");
}

template <typename T>
std::tuple<size_t, size_t> GpuImage<T>::GetPitches(const size_t width, const size_t height)
{
    /* TODO / NOTE: This is so far taken from GlassLibrary and seems to work.
    However, @mako443 is not certain if 4096 is for sure the right alignment - is this related to the QCOM page size?
    */

    cl::Device device = cl::Device::getDefault();
    auto image_pitch_alignment = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();
    int row_pitch = width * sizeof(T);  // TODO: sizeof(T) is correct, right?

    // Round up to the nearest multiple of image_pitch_alignment
    row_pitch = (row_pitch + image_pitch_alignment - 1) & ~(image_pitch_alignment - 1);

    // Round up to the nearest multiple of 4096
    // Value 4096 is reverse engineered on how cl::Image objects are constructed by reading
    // slice pitch values from cl::Image objects which owns their own memory.
    // Did not find any correct documentation on this how it should be calculated on Qualcomm GPUs.
    int slice_pitch = row_pitch * height;
    slice_pitch = (slice_pitch + 4095) & ~4095;

    return std::make_tuple(row_pitch, slice_pitch);
}

template <typename T>
size_t GpuImage<T>::GetBufferSize(const size_t width, const size_t height)
{
    auto [row_pitch, slice_pitch] = GetPitches(width, height);
    return slice_pitch * 1;
}

template <typename T>
cl::Image2D GpuImage<T>::CreateImage2dFromBuffer(GpuBuffer<T>& buffer, const size_t width, const size_t height,
                                                 cl_mem_flags flags)
{
    /// NOTE: flags needs to match what the buffer was created with, but reading them from the buffer didn't work just
    /// now.
    auto [row_pitch, slice_pitch] = GetPitches(width, height);
    if (buffer.size != slice_pitch)
        throw std::runtime_error(
            std::format("Expected a buffer of size {} as base for image, got {}.", slice_pitch, buffer.size));

    cl_image_desc image_desc;
    memset(&image_desc, 0, sizeof(image_desc));
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    image_desc.buffer = buffer.buffer().get();
    size_t pixel_size = sizeof(T);
    image_desc.image_row_pitch = row_pitch;
    image_desc.image_slice_pitch = slice_pitch;

    cl::ImageFormat format = GetClFormat();
    cl_image_format image_format;
    image_format.image_channel_order = format.image_channel_order;
    image_format.image_channel_data_type = format.image_channel_data_type;

#ifdef __APPLE__
    /*Creating an Image2D from a buffer fails on Mac, even with cl_khr_image2d_from_buffer explicitly listed.
    Therefore, I am returning a new cl::Image2D unrelated to the Buffer here. Note that this breaks having multiple
    images share the same buffer.
    */
    cl::Image2D image(gpu_context_->clContext(), flags, format, width, height);
    return image;
#else
    cl_int err;
    cl_mem image_mem =
        opencl::clCreateImage(gpu_context_->clContext().get(), flags, &image_format, &image_desc, nullptr, &err);

    if (err != CL_SUCCESS)
    {
        std::stringstream ss;
        ss << "clCreateImage() failed in CreateImage2dFromBuffer()." << "  Error code: " << std::to_string(err)
           << "  Readable error code: " << gls::clStatusToString(err) << std::endl;
        throw cl::Error(err, ss.str().c_str());
    }

    // Wrap the cl_mem object in a cl::Image2D
    return cl::Image2D(image_mem);
#endif
}

template class GpuImage<float>;
template class GpuImage<gls::pixel_fp32>;
template class GpuImage<gls::pixel_fp32_2>;
// template class GpuImage<gls::pixel_fp32_3>; /// NOTE: We explicitly leave out RGB images because they show faulty
// behaviour on Android.
template class GpuImage<gls::pixel_fp32_4>;

}  // namespace gls