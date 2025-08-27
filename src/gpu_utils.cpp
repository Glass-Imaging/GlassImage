#include "glass_image/gpu_utils.h"

namespace gls::image_utils
{
std::array<size_t, 2> GetImageShape(const cl::Image2D& image)
{
    size_t W, H;
    image.getImageInfo(CL_IMAGE_WIDTH, &W);
    image.getImageInfo(CL_IMAGE_HEIGHT, &H);
    return {W, H};
}

std::array<size_t, 3> GetImageShape(const cl::Image3D& image)
{
    size_t W, H, C;
    image.getImageInfo(CL_IMAGE_WIDTH, &W);
    image.getImageInfo(CL_IMAGE_HEIGHT, &H);
    image.getImageInfo(CL_IMAGE_DEPTH, &C);
    return {W, H, C};
}
}  // namespace gls::image_utils