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

#include "gls_ocl.hpp"
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
    std::cout << "Hello, GPU!\n";

    // Read the input file into an image object
    auto inputImage = gls::image<gls::rgba_pixel>::read_tiff_file("Assets/baboon.tiff");

    std::cout << "inputImage size: " << inputImage->width << " x " << inputImage->height << std::endl;

    // Run the OpenCL kernel
    {
        auto gpuContext = std::unique_ptr<gls::GpuContext>(new gls::OCLContext(/*programs=*/ { "blur", "blur_utils" }));

        runKernel(gpuContext.get(), *inputImage, "ocl_output.tiff");
    }

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
