#pragma once

#include <type_traits>

#include "glass_image/gpu_buffer.h"    // provides gls::GpuBuffer<T>
#include "glass_image/gpu_image.h"     // provides gls::GpuImage<T>
#include "glass_image/gpu_image_3d.h"  // provides gls::GpuImage3D<T>
#include "gls_ocl.hpp"

namespace gls::kernel_arg
{

// Pass-through for native cl types
inline cl::Buffer get_kernel_arg(const cl::Buffer& b) { return b; }
inline cl::Image2D get_kernel_arg(const cl::Image2D& i) { return i; }
inline cl::Image3D get_kernel_arg(const cl::Image3D& i) { return i; }

// Explicit overloads
template <typename T>
inline cl::Buffer get_kernel_arg(const gls::GpuBuffer<T>& buf)
{
    return buf.buffer();
}

template <typename T>
inline cl::Image2D get_kernel_arg(const gls::GpuImage<T>& img)
{
    return img.image();
}

template <typename T>
inline cl::Image3D get_kernel_arg(const gls::GpuImage3d<T>& img)
{
    return img.image();
}

// Fallback for primitive/native argument types (int, float, structs)
template <typename T>
inline std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, T> get_kernel_arg(const T& v)
{
    return v;
}

}  // namespace gls::kernel_arg