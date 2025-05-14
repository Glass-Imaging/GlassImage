// Copyright (c) 2021-2022 Glass Imaging Inc.
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

#include "cl_pipeline.h"

#include "gls_logging.h"

static const char* TAG = "CLImage Pipeline";

// Simple image processing with opencl.hpp, using cl_image to pass data to and from the GPU
int blur(gls::OpenCLContext* glsContext, const gls::cl_image_2d<gls::rgba_pixel>& input,
         gls::cl_image_2d<gls::rgba_pixel>* output) {
    try {
        // Load the shader source
        const auto blurProgram = glsContext->loadProgram("blur");

        // Bind the kernel parameters
        auto blurKernel = cl::KernelFunctor<cl::Image2D,  // input
                                            cl::Image2D   // output
                                            >(blurProgram, "blur");
        // Schedule the kernel on the GPU
        blurKernel(gls::OpenCLContext::buildEnqueueArgs(output->width, output->height), input.getImage2D(),
                   output->getImage2D());
        return 0;
    } catch (cl::Error& err) {
        gls::logging::LogError(TAG) << "Caught Exception: " << std::string(err.what()) << " - "
                                    << gls::clStatusToString(err.err()) << std::endl;
        return -1;
    }
}
