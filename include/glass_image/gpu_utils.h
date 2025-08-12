#include <array>

#include "gls_ocl.hpp"

/// TODO: This has to change! We should have gls::image::utils::SomeFunction() here and gls::image::Image/GpuImage
/// classes.
namespace gls::image_utils
{

// TODO: Even needed?
std::array<size_t, 2> GetImageShape(const cl::Image2D& image);
std::array<size_t, 3> GetImageShape(const cl::Image3D& image);

template <typename T>
cl::ImageFormat GetClFormat()
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
std::tuple<size_t, size_t> GetPitches(const size_t width, const size_t height)
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
size_t GetBufferSize(const size_t width, const size_t height, const size_t depth = 1)
{
    auto [row_pitch, slice_pitch] = GetPitches<T>(width, height);
    return slice_pitch * depth;
}

}  // namespace gls::image_utils