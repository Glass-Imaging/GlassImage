#ifndef gls_gpu_image_transform_hpp
#define gls_gpu_image_transform_hpp

#include "gls_gpu_image.hpp"
#include <utility>

namespace gls {

/**
 * @brief Interface for GPU-based image transformations that support resource preallocation
 * 
 * This interface defines the contract for image processing operations that can benefit
 * from pre-allocated GPU resources. The template parameters allow for compile-time
 * type safety and explicit format specifications.
 * 
 * @tparam InputType The pixel type for input images (e.g., rgba_pixel_fp16)
 * @tparam OutputType The pixel type for output images, defaults to same as input
 */
template<typename InputType, typename OutputType = InputType>
class GpuImageTransform {
protected:
    // Protected member to store the GPU context
    GpuContext* _context;

public:
    GpuImageTransform(GpuContext* context) : _context(context) {}
    virtual ~GpuImageTransform() = default;

    /**
     * @brief Calculate the output dimensions for a given input size
     * 
     * By default, returns the same size as the input. Override this method
     * if your transform changes the image dimensions (e.g., scaling, cropping).
     * 
     * @param inSize The dimensions of the input image
     * @return gls::size The dimensions of the output image that will be produced
     */
    virtual gls::size getOutSize(const gls::size& inSize) const {
        return inSize;  // Default: output same size as input
    }

    /**
     * @brief Create a new output image with the correct format and dimensions
     * 
     * By default, creates a new gpu_image of OutputType with dimensions from getOutSize.
     * Override this method if your transform needs special output image configuration.
     * 
     * @param inSize The dimensions of the input image that will be processed
     * @return unique_ptr to a newly allocated image suitable for output
     */
    virtual typename gpu_image<OutputType>::unique_ptr createOutImage(const gls::size& inSize) const {
        return _context->template new_gpu_image_2d<OutputType>(getOutSize(inSize));
    }

    /**
     * @brief Preallocate GPU resources for processing images of the specified size
     * 
     * By default, does nothing since most transforms don't need preallocation.
     * Override this method if your transform needs to prepare resources before processing.
     * 
     * @param inSize The dimensions of input images that will be processed
     */
    virtual void preallocate(const gls::size& inSize) {
        // Default: no preallocation needed
    }

    /**
     * @brief Check if resources are currently preallocated for the specified size
     * 
     * By default, returns true since most transforms don't need preallocation.
     * Override this method if your transform uses preallocated resources.
     * 
     * @param inSize The dimensions to check for preallocation
     * @return true if resources are preallocated for the given size
     * @return false if resources need to be preallocated before processing
     */
    virtual bool isPreallocated(const gls::size& inSize) const {
        return true;  // Default: no preallocation needed
    }

    /**
     * @brief Process an input image and write the result to an output image
     * 
     * The input and output types are guaranteed to match InputType and OutputType
     * respectively at compile time. The output image must have dimensions matching
     * those returned by getOutSize() for the input image's dimensions.
     * 
     * @param input Source image to process
     * @param output Destination for the processed result
     * @return true if processing succeeded, false otherwise
     */
    virtual bool submit(const gpu_image<InputType>& input, 
                      gpu_image<OutputType>& output) = 0;

    /**
     * @brief Process an input image and return the result as a new image
     * 
     * Convenience method that creates an appropriate output image and processes
     * the input. The caller takes ownership of the returned image. Input and output
     * types are guaranteed to match InputType and OutputType respectively.
     * 
     * @param input Source image to process
     * @return std::pair<bool, unique_ptr<gpu_image<OutputType>>> Success flag and the processed result
     */
    virtual std::pair<bool, typename gpu_image<OutputType>::unique_ptr> 
    submit(const gpu_image<InputType>& input) {
        auto output = createOutImage(input.size());
        bool success = submit(input, *output);
        return std::make_pair(success, std::move(output));
    }

    /**
     * @brief Alias for submit() that processes an input image and writes to an output image
     *
     * This can make the call to a transform more compact with transform(input, output).
     * 
     * @param input Source image to process
     * @param output Destination for the processed result
     * @return true if processing succeeded, false otherwise
     */
    bool operator()(const gpu_image<InputType>& input, 
                   gpu_image<OutputType>& output) {
        return submit(input, output);
    }

    /**
     * @brief Alias for submit() that processes an input image and returns a new image
     * 
     * This can make the call to a transform more compact with transform(input, output).
     * 
     * @param input Source image to process
     * @return std::pair<bool, unique_ptr<gpu_image<OutputType>>> Success flag and the processed result
     */
    std::pair<bool, typename gpu_image<OutputType>::unique_ptr> 
    operator()(const gpu_image<InputType>& input) {
        return submit(input);
    }

    /**
     * @brief Release any preallocated resources
     * 
     * By default, does nothing since most transforms don't need resource cleanup.
     * Override this method if your transform needs to free resources explicitly.
     */
    virtual void releaseResources() { }
};

} // namespace gls

#endif /* gls_gpu_image_transform_hpp */ 
