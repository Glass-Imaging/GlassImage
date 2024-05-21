// Copyright (c) 2021-2023 Glass Imaging Inc.
// Author: Fabio Riccardi <fabio@glass-imaging.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <complex>
#include "gls_ocl.hpp"
#include "cnpy.h"

#ifdef __APPLE__
#include "gls_mtl.hpp"
#endif

struct GpuBlurKernel {
    gls::Kernel<
        gls::gpu_image<gls::rgba_pixel>,      // inputImage
        gls::gpu_image<gls::rgba_pixel>,      // outputImage
        gls::gpu_buffer<int>
    > blur;

    gls::gpu_buffer<int>::unique_ptr kernelSizeBuffer;

    GpuBlurKernel(gls::GpuContext* context) : blur(context, "blur"), kernelSizeBuffer(context->new_buffer(15, /*readOnly=*/ true)) { }

    void operator() (gls::GpuContext* context,
                     const gls::gpu_image<gls::rgba_pixel>& inputImage,
                     gls::gpu_image<gls::rgba_pixel>* outputImage) const {
        blur(context, /*gridSize=*/ inputImage.size(), inputImage, *outputImage, *kernelSizeBuffer);
    }
};

void runKernel(gls::GpuContext* gpuContext, const gls::image<gls::rgba_pixel>& inputImage, const std::string& outputFile) {
    auto gpuInputImage = gpuContext->new_gpu_image_2d<gls::rgba_pixel>(inputImage);
    auto gpuOutputImage = gpuContext->new_gpu_image_2d<gls::rgba_pixel>(gpuInputImage->width, gpuInputImage->height);

    GpuBlurKernel blur(gpuContext);

    blur(gpuContext, *gpuInputImage, gpuOutputImage.get());

    gpuContext->waitForCompletion();

    // Use OpenCL's memory mapping for zero-copy image Output
    auto outputImage = gpuOutputImage->mapImage();
    outputImage.write_tiff_file(outputFile.c_str());
}

int main(int argc, const char * argv[]) {

    std::cout << "Hello from C++!\n";

    //load from npy file
    cnpy::NpyArray raw_data = cnpy::npy_load(argv[1]);

    std::cout << "The byte of pixel size is:" << raw_data.word_size << std::endl;
    std::cout << "The size of Npy array is:" << raw_data.shape[0] <<" x " << raw_data.shape[1] << std::endl;

    if (word_size == sizeof(uint16_t)):
    // two bytes for each pixel, uint_16 type
        uint16_t * loaded_data = raw_data.data<uint16_t>();
    else:
    // two bytes for each pixel, uint_8 type
        uint8_t * loaded_data = raw_data.data<uint8_t>();

//    auto outputImage = gls::image<gls::rgb_pixel>(raw_data.shape[1], raw_data.shape[0]);
    auto outputImage = gls::image<gls::luma_pixel>(raw_data.shape[1], raw_data.shape[0]);

    // import the QNN module and run the OF estimation
    // Example here reduce data from 12bit to 8bit to feed into network
    for(int y = 0; y < raw_data.shape[0]; y++) {
        for(int x = 0; x < raw_data.shape[1]; x++) {
            outputImage[y][x] = loaded_data[y*raw_data.shape[1] + x] / 16;
        }
    }

    outputImage.write_png_file("output_test_results.png");

    // Run the OpenCL kernel - Needs Debugging
    // {
    //     auto gpuContext = std::unique_ptr<gls::GpuContext>(new gls::OCLContext(/*programs=*/ { "blur", "blur_utils" }));
    //     runKernel(gpuContext.get(), *inputImage, "ocl_output.tiff");
    // }


#ifdef __APPLE__
    // Run the Metal kernel
    {
        auto allMetalDevices = NS::TransferPtr(MTL::CopyAllDevices());
        auto metalDevice = NS::RetainPtr(allMetalDevices->object<MTL::Device>(0));
        auto gpuContext = std::unique_ptr<gls::GpuContext>(new gls::MetalContext(metalDevice));

        runKernel(gpuContext.get(), *inputImage, "metal_output.tiff");
    }
#endif
    return 0;
}
