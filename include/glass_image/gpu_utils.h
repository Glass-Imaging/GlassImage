#include <array>

#include "gls_ocl.hpp"

/// TODO: This has to change! We should have gls::image::utils::SomeFunction() here and gls::image::Image/GpuImage
/// classes.
namespace gls::image_utils
{

// TODO: Even needed?
std::array<size_t, 2> GetImageShape(const cl::Image2D& image);
std::array<size_t, 3> GetImageShape(const cl::Image3D& image);
}  // namespace gls::image_utils