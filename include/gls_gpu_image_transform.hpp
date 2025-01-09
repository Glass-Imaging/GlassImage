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
public:
    virtual ~GpuImageTransform() = default;

    /**
     * @brief Calculate the output dimensions for a given input size
     * 
     * @param inSize The dimensions of the input image
     * @return gls::size The dimensions of the output image that will be produced
     */
    virtual gls::size getOutSize(const gls::size& inSize) const = 0;

    /**
     * @brief Create a new output image with the correct format and dimensions
     * 
     * Creates and returns a new gpu_image configured with the appropriate format
     * and dimensions for this transform's output. The caller takes ownership
     * of the returned image. The output format is guaranteed to match OutputType.
     * 
     * @param inSize The dimensions of the input image that will be processed
     * @return unique_ptr to a newly allocated image suitable for output
     */
    virtual typename gpu_image<OutputType>::unique_ptr createOutImage(const gls::size& inSize) const = 0;

    /**
     * @brief Preallocate GPU resources for processing images of the specified size
     * 
     * This method allows implementations to allocate buffers, create pipeline states,
     * and prepare any other resources needed for processing images of the given size.
     * 
     * @param inSize The dimensions of input images that will be processed
     */
    virtual void preallocate(const gls::size& inSize) = 0;

    /**
     * @brief Check if resources are currently preallocated for the specified size
     * 
     * @param inSize The dimensions to check for preallocation
     * @return true if resources are preallocated for the given size
     * @return false if resources need to be preallocated before processing
     */
    virtual bool isPreallocated(const gls::size& inSize) const = 0;

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
    virtual bool operator()(const gpu_image<InputType>& input, 
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
    operator()(const gpu_image<InputType>& input) {
        auto output = createOutImage(input.size());
        bool success = operator()(input, *output);
        return std::make_pair(success, std::move(output));
    }

    /**
     * @brief Release any preallocated resources
     * 
     * Optional method to free GPU resources. Default implementation does nothing.
     * Implementations should override this if they need explicit cleanup.
     */
    virtual void releaseResources() { }
};

} // namespace gls

#endif /* gls_gpu_image_transform_hpp */ 