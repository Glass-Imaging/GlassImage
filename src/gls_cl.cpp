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

#include <fstream>
#include <iostream>
#include <map>

#include "gls_cl.hpp"
#include "gls_logging.h"

namespace gls {

static const char* TAG = "CLImage";

#ifndef __ANDROID__  // On Apple and Linux

OpenCLContext::OpenCLContext(const std::string& shadersRootPath, bool quiet) : _shadersRootPath(shadersRootPath) {
    _clContext = cl::Context::getDefault();

    std::vector<cl::Device> devices = _clContext.getInfo<CL_CONTEXT_DEVICES>();

    // Macs have multiple GPUs, select the one with most compute units
    int max_compute_units = 0;
    cl::Device best_device;
    for (const auto& d : devices) {
        int device_compute_units = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        if (device_compute_units > max_compute_units) {
            max_compute_units = device_compute_units;
            best_device = d;
        }
    }
    cl::Device::setDefault(best_device);

    if (!quiet) {
        cl::Device d = cl::Device::getDefault();
        LOG_INFO(TAG) << "OpenCL Default Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
        LOG_INFO(TAG) << "- Device Version: " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Driver Version: " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- OpenCL C Version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
    }
}

#else  // Android

OpenCLContext::OpenCLContext(const std::string& shadersRootPath, bool quiet) : _shadersRootPath(shadersRootPath) {
    // Load libOpenCL
    CL_WRAPPER_NS::bindOpenCLLibrary();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform;
    for (auto& p : platforms) {
        std::string version = p.getInfo<CL_PLATFORM_VERSION>();
        if (version.find("OpenCL 2.") != std::string::npos || version.find("OpenCL 3.") != std::string::npos) {
            platform = p;
        }
    }
    if (platform() == nullptr) {
        throw cl::Error(-1, "No OpenCL 2.0 platform found.");
    }

    cl::Platform defaultPlatform = cl::Platform::setDefault(platform);
    if (defaultPlatform != platform) {
        throw cl::Error(-1, "Error setting default platform.");
    }

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};
    cl::Context context(CL_DEVICE_TYPE_ALL, properties);

    cl::Device d = cl::Device::getDefault();
    if (!quiet) {
        LOG_INFO(TAG) << "- Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
        LOG_INFO(TAG) << "- Device Version: " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Driver Version: " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- OpenCL C Version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        LOG_INFO(TAG) << "- Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        LOG_INFO(TAG) << "- CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
    }

    // opencl.hpp relies on a default context
    cl::Context::setDefault(context);
    _clContext = cl::Context::getDefault();
}

#endif

std::string OpenCLContext::OpenCLSource(const std::string& shaderName) {
#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
    return cl_shaders[shaderName];
#else
    std::ifstream file(_shadersRootPath + "OpenCL/" + shaderName, std::ios::in | std::ios::ate);
    if (file.is_open()) {
        std::streampos size = file.tellg();
        std::vector<char> memblock((int)size);
        file.seekg(0, std::ios::beg);
        file.read(memblock.data(), size);
        file.close();
        return std::string(memblock.data(), memblock.data() + size);
    }
    return "";
#endif
}

std::vector<unsigned char> OpenCLContext::OpenCLBinary(const std::string& shaderName) {
#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
    return cl_bytecode[shaderName];
#else
    std::ifstream file(_shadersRootPath + "OpenCLBinaries/" + shaderName,
                       std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streampos size = file.tellg();
        std::vector<unsigned char> memblock((int)size);
        file.seekg(0, std::ios::beg);
        file.read((char*)memblock.data(), size);
        file.close();
        return memblock;
    }
    return std::vector<unsigned char>();
#endif
}

// Static
int OpenCLContext::saveBinaryFile(const std::string& path, const std::vector<unsigned char>& binary) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
        file.write((char*)binary.data(), binary.size());
        file.close();
        return 0;
    }
    LOG_ERROR(TAG) << "Couldn't open file " << path << std::endl;
    return -1;
}

// Static
void OpenCLContext::handleProgramException(const cl::BuildError& e) {
    LOG_ERROR(TAG) << "OpenCL Build Error - " << e.what() << ": " << clStatusToString(e.err()) << std::endl;
    // Print build info for all devices
    for (auto& pair : e.getBuildLog()) {
        LOG_ERROR(TAG) << pair.second << std::endl;
    }
}

// NOTE: using -cl-fast-relaxed-math actually reduces precision on macOS, it also doesn't seem to increase performance

#ifdef __APPLE__
static const char* cl_options = "-cl-std=CL1.2 -cl-single-precision-constant";
#else
static const char* cl_options = "-cl-std=CL2.0 -Werror -cl-single-precision-constant -I OpenCL";
#endif

cl::Program OpenCLContext::loadProgram(const std::string& programName, const std::string& shadersRootPath) {
    cl::Program program = _program_cache[programName];
    if (program()) {
        return program;
    }

    try {
        cl::Context context = clContext();
        cl::Device device = cl::Device::getDefault();

#if (defined(__ANDROID__) && defined(NDEBUG)) || (defined(__APPLE__) && !defined(__aarch64__))
        std::vector<unsigned char> binary = OpenCLBinary(programName + ".o");

        if (!binary.empty()) {
            program = cl::Program(context, {device}, {binary});
        } else
#endif
        {
            program = cl::Program(OpenCLSource(programName + ".cl"));
        }
        program.build(device, cl_options);
        _program_cache[programName] = program;
        return program;
    } catch (const cl::BuildError& e) {
        handleProgramException(e);
        return cl::Program();
    }
}

// Static
int OpenCLContext::buildProgram(cl::Program& program) {
    try {
        program.build(cl_options);
        for (auto& pair : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()) {
            if (!pair.second.empty() && pair.second != "Pass") {
                LOG_INFO(TAG) << "OpenCL Build: " << pair.second << std::endl;
            }
        }
    } catch (const cl::BuildError& e) {
        handleProgramException(e);
        return -1;
    }
    return 0;
}

// Compute a list of divisors in the range [1..32]
static std::vector<int> computeDivisors(const size_t val) {
    std::vector<int> divisors;
    int divisor = 32;
    while (divisor >= 1) {
        if (val % divisor == 0) {
            divisors.push_back(divisor);
        }
        divisor /= 2;
    }
    return divisors;
}

// Compute the squarest workgroup of size <= max_workgroup_size
// Static
cl::NDRange OpenCLContext::computeWorkGroupSizes(size_t width, size_t height) {
    std::vector<int> width_divisors = computeDivisors(width);
    std::vector<int> height_divisors = computeDivisors(height);

    cl::Device d = cl::Device::getDefault();
    const size_t max_workgroup_size = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    int width_divisor = 1;
    int height_divisor = 1;
    while (width_divisor * height_divisor <= max_workgroup_size &&
           (!width_divisors.empty() || !height_divisors.empty())) {
        if (!width_divisors.empty()) {
            int new_width_divisor = width_divisors.back();
            width_divisors.pop_back();
            if (new_width_divisor * height_divisor > max_workgroup_size) {
                break;
            } else {
                width_divisor = new_width_divisor;
            }
        }
        if (!height_divisors.empty()) {
            int new_height_divisor = height_divisors.back();
            height_divisors.pop_back();
            if (new_height_divisor * width_divisor > max_workgroup_size) {
                break;
            } else {
                height_divisor = new_height_divisor;
            }
        }
    }
    //    LOG_INFO(TAG) << "computeWorkGroupSizes for " << width << ", " << height << ": "
    //                  << width_divisor << ", " << height_divisor
    //                  << " (" << width_divisor * height_divisor << ") of " <<  max_workgroup_size << std::endl;
    return cl::NDRange(width_divisor, height_divisor);
}

}  // namespace gls
