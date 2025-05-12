//
//  main.cpp
//  BlurImageTest
//
//  Created by Fabio Riccardi on 2/10/23.
//

#include <string>

#include "cl_pipeline.h"
#include "gls_cl.hpp"
#include "gls_cl_image.hpp"
#include "gls_logging.h"

static const char* TAG = "CLImage Test";

int main(int argc, const char* argv[]) {
    printf("Hello CLImage!\n");

    if (argc > 1) {
        // Initialize the OpenCL environment and get the context
        gls::OpenCLContext glsContext("");
        auto clContext = glsContext.clContext();

        // Read the input file into an image object
        auto inputImage = gls::image<gls::rgba_pixel>::read_tiff_file(argv[1]);

        gls::logging::LogDebug(TAG) << "inputImage size: " << inputImage->width << " x " << inputImage->height
                                    << std::endl;

        // Load image data in OpenCL image texture
        gls::cl_image_2d<gls::rgba_pixel> clInputImage(clContext, *inputImage);

        // Output Image from OpenCL processing
        gls::cl_image_2d<gls::rgba_pixel> clOutputImage(clContext, clInputImage.width, clInputImage.height);

        // Execute OpenCL Blur algorithm
        if (blur(&glsContext, clInputImage, &clOutputImage) == 0) {
            gls::logging::LogDebug(TAG) << "All done with Blur" << std::endl;
        } else {
            gls::logging::LogError(TAG) << "Something wrong with the Blur." << std::endl;
        }

        // Use OpenCL's memory mapping for zero-copy image Output
        auto outputImage = clOutputImage.mapImage();
        outputImage.write_tiff_file("output.tiff");
        clOutputImage.unmapImage(outputImage);
    }

    return 0;
}
