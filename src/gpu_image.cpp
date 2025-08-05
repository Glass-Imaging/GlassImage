#include "glass_image/gpu_image.h"

#include <format>
#include <stdexcept>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_utils.h"
#include "gls_image.hpp"

namespace gu = gls::image_utils;

namespace gls
{

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const std::array<size_t, 2> shape,
                      cl_mem_flags flags)
    : gpu_context_(gpu_context), shape_(shape)
{
    image_ = cl::Image2D(gpu_context->clContext(), flags, GetClFormat(), shape[0], shape[1]);
}

template <typename T>
GpuImage<T>::GpuImage(std::shared_ptr<gls::OCLContext> gpu_context, const gls::image<T>& image, cl_mem_flags flags)
    : gpu_context_(gpu_context), shape_({(size_t)image.width, (size_t)image.height})
{
    image_ = cl::Image2D(gpu_context->clContext(), flags, GetClFormat(), shape_[0], shape_[1]);
    CopyFrom(image).wait();
}

template <typename T>
gls::image<T> GpuImage<T>::ToImage(std::optional<cl::CommandQueue> queue, const std::vector<cl::Event>& events)
{
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    auto [width, height] = shape_;
    gls::image<T> host_image(width, height);
    _queue.enqueueReadImage(image_, CL_TRUE, {0, 0, 0}, {width, height, 1}, 0, 0, host_image.pixels().data(), &events);
    return host_image;
}

template <typename T>
cl::Event GpuImage<T>::CopyFrom(const gls::image<T>& image, std::optional<cl::CommandQueue> queue,
                                const std::vector<cl::Event>& events)
{
    if (image.width != shape_[0] || image.height != shape_[1])
        throw std::runtime_error(std::format("LoadImage() expected image of size {}x{}, got {}x{}.", shape_[0],
                                             shape_[1], image.width, image.height));

    /// TODO: Potentially consider strides! Hopefully enough to set row pitch.
    cl::CommandQueue _queue = queue.value_or(gpu_context_->clCommandQueue());
    cl::Event event;
    const size_t row_pitch = image.stride * sizeof(T);
    _queue.enqueueWriteImage(image_, CL_FALSE, {0, 0, 0}, {shape_[0], shape_[1], 1}, row_pitch,
                             row_pitch * image.height, image.pixels().data(), &events, &event);
    return event;
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

template class GpuImage<float>;
template class GpuImage<gls::pixel_fp32>;
template class GpuImage<gls::pixel_fp32_2>;
// template class GpuImage<gls::pixel_fp32_3>; /// NOTE: We explicitly leave out RGB images because they show faulty
// behaviour on Android.
template class GpuImage<gls::pixel_fp32_4>;

}  // namespace gls