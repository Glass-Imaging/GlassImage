#ifndef gls_gpu_transform_hpp
#define gls_gpu_transform_hpp

#include "gls_gpu_image.hpp"
#include <utility>

namespace gls {

/**
 * @brief Pure virtual interface for GPU-based transformations
 * 
 * This interface defines the contract for GPU operations that transform
 * one type of resource into another. The template parameters allow for 
 * compile-time type safety and explicit format specifications.
 * 
 * @tparam InputType The type of input resource
 * @tparam OutputType The type of output resource, defaults to same as input
 */
template<typename InputType, typename OutputType = InputType>
class GpuTransform {
public:
    virtual ~GpuTransform() = default;

    /**
     * @brief Calculate the output dimensions for a given input size
     * 
     * @param inSize The dimensions of the input resource
     * @return gls::size The dimensions of the output resource that will be produced
     */
    virtual gls::size getOutSize(const gls::size& inSize) const = 0;

    /**
     * @brief Preallocate resources for processing inputs of the specified size
     * 
     * @param inSize The dimensions of inputs that will be processed
     * @return true if preallocation was successful
     * @return false if preallocation failed
     */
    virtual bool preallocate(const gls::size& inSize) = 0;

    /**
     * @brief Check if resources are currently preallocated for the specified size
     * 
     * @param inSize The dimensions to check for preallocation
     * @return true if resources are preallocated for the given size
     * @return false if resources need to be preallocated before processing
     */
    virtual bool isPreallocated(const gls::size& inSize) const = 0;

    /**
     * @brief Process an input resource and write the result to an output resource
     * 
     * The input and output types are guaranteed to match InputType and OutputType
     * respectively at compile time. The output must have dimensions matching
     * those returned by getOutSize() for the input's dimensions.
     * 
     * @param input Source resource to process
     * @param output Destination for the processed result
     * @return true if processing succeeded, false otherwise
     */
    virtual bool submit(const InputType& input, OutputType& output) = 0;

    /**
     * @brief Release any preallocated resources
     * 
     * @return true if cleanup succeeded, false otherwise
     */
    virtual bool releaseResources() = 0;

    /**
     * @brief Alias for submit() that processes an input and writes to an output
     *
     * This can make the call to a transform more compact with transform(input, output).
     * 
     * @param input Source resource to process
     * @param output Destination for the processed result
     * @return true if processing succeeded, false otherwise
     */
    bool operator()(const InputType& input, OutputType& output) {
        return submit(input, output);
    }
};

/**
 * @brief Base class for GPU image transformations that implement the GpuTransform interface
 * 
 * 
 * @tparam PixelInputType The pixel type for input images (e.g., rgba_pixel_fp16)
 * @tparam PixelOutputType The pixel type for output images, defaults to same as input
 */
template<typename PixelInputType, typename PixelOutputType = PixelInputType>
class GpuImageTransform : public GpuTransform<gpu_image<PixelInputType>, gpu_image<PixelOutputType> > {
protected:
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
    virtual gls::size getOutSize(const gls::size& inSize) const override {
        return inSize;  // Default: output same size as input
    }

    /**
     * @brief Preallocate GPU resources for processing images of the specified size
     * 
     * By default, does nothing since most transforms don't need preallocation.
     * Override this method if your transform needs to prepare resources before processing.
     * 
     * @param inSize The dimensions of input images that will be processed
     * @return true if preallocation was successfull
     * @return false if preallocation failed
     */
    virtual bool preallocate(const gls::size& inSize) override {
        return true;  // Default: no preallocation needed
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
    virtual bool isPreallocated(const gls::size& inSize) const override {
        return true;  // Default: no preallocation needed
    }

    virtual bool releaseResources() override {
        return true;  // Default: no cleanup needed
    }

    /**
     * @brief Process an input image and write the result to an output image
     * 
     * This is the core implementation that derived classes must provide.
     * 
     * @param input Source image to process
     * @param output Destination for the processed result
     * @return true if processing succeeded, false otherwise
     */
    virtual bool submit(const gpu_image<PixelInputType>& input, 
                       gpu_image<PixelOutputType>& output) override = 0;

    /**
     * @brief Create a new output image with the correct format and dimensions
     *
     * By default, creates a new gpu_image of PixelOutputType with dimensions from getOutSize.
     * Override this method if your transform needs special output image configuration.
     *
     * @param inSize The dimensions of the input image that will be processed
     * @return unique_ptr to a newly allocated image suitable for output
     */
    virtual typename gpu_image<PixelOutputType>::unique_ptr createOutImage(const gls::size& inSize) const {
        return _context->template new_gpu_image_2d<PixelOutputType>(getOutSize(inSize));
    }
};

} // namespace gls

#endif /* gls_gpu_transform_hpp */ 
