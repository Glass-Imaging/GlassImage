#pragma once

#include <cxxabi.h>

#include <string>

#include "glass_image/gpu_buffer.h"
#include "glass_image/gpu_image.h"
#include "glass_image/gpu_image_3d.h"
#include "gls_ocl.hpp"

namespace gls
{

/// TODO: Make abstract
class GpuKernel
{
    /* We create GpuKernel as a shallow wrapper around cl::Kernel for increased safety and clear typing.

    We give GpuKernel its own class to potentially add utility functions like "scale factor".
    */
   public:
    GpuKernel(std::shared_ptr<gls::OCLContext> gpu_context, const std::string name);

    /// TODO: @mako443 should explain the C++ 17 template magic.

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

    /* We specify 3 specific templates for GpuBuffer, GpuImage, GpuImage3d such that they can be passed in directly. */
    template <typename T>
    void SetArg(const size_t index, const GpuBuffer<T>& buffer)
    {
        try
        {
            kernel_.setArg(index, buffer.buffer());
        }
        catch (cl::Error& e)
        {
            throw std::runtime_error(
                std::format("Failed setting {} arg {} with GpuBuffer<{}>.", name_, index, typeid(T).name()));
        }
    }

    template <typename T>
    void SetArg(const size_t index, const GpuImage<T>& image)
    {
        try
        {
            kernel_.setArg(index, image.image());
        }
        catch (cl::Error& e)
        {
            throw std::runtime_error(
                std::format("Failed setting {} arg {} with GpuImage<{}>.", name_, index, typeid(T).name()));
        }
    }

    template <typename T>
    void SetArg(const size_t index, const GpuImage3d<T>& image)
    {
        try
        {
            kernel_.setArg(index, image.image());
        }
        catch (cl::Error& e)
        {
            throw std::runtime_error(
                std::format("Failed setting {} arg {} with GpuImage3d<{}>.", name_, index, typeid(T).name()));
        }
    }

    std::shared_ptr<gls::OCLContext> gpu_context_;

   protected:
    cl::Kernel kernel_;
    const std::string name_;
};

}  // namespace gls