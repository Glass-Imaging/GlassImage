#pragma once

#include <cxxabi.h>

#include <optional>
#include <span>
#include <string>
#include <vector>

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
        std::cout << "Setting arg " << index << " with " << typeid(T).name() << std::endl;
        try
        {
            kernel_.setArg(index, arg);
        }
        catch (cl::Error& e)
        {
            // Quick GPT code to go from weird C++ type names like "i" for int or "N2cl6BufferE" for cl::Buffer to the
            // human-readable names. Hope this is stable :)
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