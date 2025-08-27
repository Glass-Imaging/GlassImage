#include <array>

#include "gls_image.hpp"
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
    else if constexpr (std::is_same_v<T, float16_t>)
        return cl::ImageFormat(CL_R, CL_HALF_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp16>)
        return cl::ImageFormat(CL_R, CL_HALF_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp16_2>)
        return cl::ImageFormat(CL_RG, CL_HALF_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp16_4>)
        return cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp32>)
        return cl::ImageFormat(CL_R, CL_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp32_2>)
        return cl::ImageFormat(CL_RG, CL_FLOAT);
    else if constexpr (std::is_same_v<T, gls::pixel_fp32_4>)
        return cl::ImageFormat(CL_RGBA, CL_FLOAT);
    else if constexpr (std::is_same_v<T, gls::luma_pixel_16>)
        return cl::ImageFormat(CL_R, CL_UNSIGNED_INT16);
    else
        throw std::runtime_error("Unsupported pixel type for GpuImage::GetClFormat()");
}

/// Get the optimal row pitch for an image with the given width in pixels, adhering to the device constraints and
/// potentially padded to a non-power of 2.
/// @param width Width in pixels
/// @return Optimal row pitch in pixels
template <typename T>
size_t GetBestRowPitch(size_t width)
{
    /* According to @mako443's understanding, the row pitch has to be a multiple of CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
       both in pixels. Furthermore, experiments on the Adreno GPU of Xiaomi 15 Ultra showed that read & write speeds can
       be ~2x faster if the row pitch is *not* a power of 2 - this is fully counter-intuitive. Therefore, I am
       experimentally padding power-of-2 pitches to the next valid pitch size based on the GLASS_IMAGE_PAD_POWER2_IMAGES
       preproc directive.
    */

    cl::Device device = cl::Device::getDefault();
    cl_uint image_pitch_alignment = device.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();  // In pixels

    // Round up width to the next multiple of image_pitch_alignment, all in pixels.
    width = ((width + image_pitch_alignment - 1) / image_pitch_alignment) * image_pitch_alignment;

#if GLASS_IMAGE_PAD_POWER2_IMAGES
    if ((width & (width - 1)) == 0) width += image_pitch_alignment;
#endif

    return width;
}

}  // namespace gls::image_utils