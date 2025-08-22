#pragma once

#include <cxxabi.h>

#include <string>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "glass_image/gpu_image_3d.h"
#include "gls_ocl.hpp"

namespace gls
{

namespace detail
{
// Pass-through for native cl types
inline cl::Buffer GetKernelArg(const cl::Buffer& b) { return b; }
inline cl::Image2D GetKernelArg(const cl::Image2D& i) { return i; }
inline cl::Image3D GetKernelArg(const cl::Image3D& i) { return i; }

// Explicit overloads
template <typename T>
inline cl::Buffer& GetKernelArg(const gls::GpuBuffer<T>& buf)
{
    return buf.buffer();
}

template <typename T>
inline cl::Image2D GetKernelArg(const gls::GpuImage<T>& img)
{
    return img.image();
}

template <typename T>
inline cl::Image3D GetKernelArg(const gls::GpuImage3d<T>& img)
{
    return img.image();
}

// Fallback for primitive/native argument types (int, float, structs)
template <typename T>
inline std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, T> GetKernelArg(const T& v)
{
    return v;
}
}  // namespace detail

/// TODO: Make abstract
class GpuKernel
{
    /* We create GpuKernel as a shallow wrapper around cl::Kernel for increased safety and clear typing.

    We give GpuKernel its own class to potentially add utility functions like "scale factor".
    */
   public:
    GpuKernel(std::shared_ptr<gls::OCLContext> gpu_context, const std::string name);

    // clang-format off
    /* TODO: @mako443 will explain the SetArg template magic.
    Is this worth it for us? It is harder to understand and creates potentially less clear error messages but cuts down something like
        SetArg(0, some_buffer);
        SetArg(1, some_image);
    down to 
        SetArgs(some_buffer, some_image);
    in the kernel wrappers.
    */
    // clang-format on

    /// NOTE: Quick warning, currently you have to call this like SetArgs(gpu_buffer.buffer(), gpu_image.image(),
    /// some_integer); This could potentially be updated.
    template <typename... Args>
    void SetArgs(Args&&... args)
    {
        size_t index = 0;
        (SetArg(index++, std::forward<Args>(args)), ...);  // C++17 fold expression
    }

    template <typename T>
    void SetArg(const size_t index, T arg)
    {
        try
        {
            // kernel_.setArg(index, detail::GetKernelArg(arg));  // Currently gives some weird buffer copy error!
            kernel_.setArg(index, arg);
        }
        catch (cl::Error& e)
        {
            int status;
            char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
            std::string demangled_str = (status == 0) ? demangled : typeid(T).name();
            free(demangled);
            throw std::runtime_error(
                std::format("Failed setting {} arg {} with type {}.", name_, index, demangled_str));
        }
    }

    std::shared_ptr<gls::OCLContext> gpu_context_;

   protected:
    cl::Kernel kernel_;
    const std::string name_;
};

}  // namespace gls