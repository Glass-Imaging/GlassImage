//
//  gls_ocl.h
//  GlassCamera
//
//  Created by Fabio Riccardi on 8/7/23.
//

#ifndef gls_ocl_h
#define gls_ocl_h

#include <cmath>
#include <map>
#include <fstream>
#include <iostream>

#define CL_HPP_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR true

#include <OpenCL/cl_ext.h>

#include "CL/opencl.hpp"

#elif __ANDROID__

#define USE_ASSET_MANAGER // Use asset manager to load OpenCL kernels
#define USE_TEXT_SHADERS

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

// clang-format off

// Include cl_icd_wrapper.h before <CL/*>
#include "gls_icd_wrapper.h"

#include <CL/cl_ext.h>
#include <CL/opencl.hpp>
#include <android/log.h>

// clang-format on

#elif __linux__

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl_ext.h>

#include <CL/opencl.hpp>

#endif

#include "gls_gpu_image.hpp"
#include "gls_ocl_image.hpp"

namespace gls {

class OCLCommandEncoder : public GpuCommandEncoder {
    cl::Kernel* _kernel;

public:
    OCLCommandEncoder(cl::Kernel& kernel) : _kernel(&kernel) { }

    virtual ~OCLCommandEncoder() { }

    virtual void setBytes(const void* parameter, size_t parameter_size, unsigned index) override {
        _kernel->setArg(index, parameter_size, parameter);
    }

    virtual void setBuffer(const gls::buffer& buffer, unsigned index) override {
        if (const ocl_buffer* b = dynamic_cast<const ocl_buffer*>(buffer())) {
            _kernel->setArg(index, b->buffer());
        } else {
            throw std::runtime_error("Unexpected buffer type.");
        }
    }

    virtual void setTexture(const gls::texture& texture, unsigned index) override {
        if (const ocl_texture* t = dynamic_cast<const ocl_texture*>(texture())) {
            _kernel->setArg(index, t->image());
        } else {
            throw std::runtime_error("Unexpected buffer type.");
        }
    }
};

#ifndef OPENCL_HEADERS_PATH
#define OPENCL_HEADERS_PATH ""
#endif

#ifdef __APPLE__
static const char* cl_options = "-cl-std=CL1.2 -cl-single-precision-constant -I " OPENCL_HEADERS_PATH "OpenCL";
#else
static const char* cl_options = "-cl-std=CL2.0 -Werror -cl-single-precision-constant -I " OPENCL_HEADERS_PATH "OpenCL";
#endif

class OCLContext : public GpuContext {
    cl::Context _clContext;
    cl::Program _program;
    const std::string _shadersRootPath;

#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
        std::map<std::string, std::string> cl_shaders;
         std::map<std::string, std::vector<unsigned char>> cl_bytecode;
#endif

public:
    OCLContext(const std::vector<std::string>& programs, const std::string& shadersRootPath = "") : _shadersRootPath(shadersRootPath) {
#if __ANDROID__
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
        std::cout << "- Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "- Device Version: " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
        std::cout << "- Driver Version: " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        std::cout << "- OpenCL C Version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        std::cout << "- Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "- CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        std::cout << "- CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;

        // opencl.hpp relies on a default context
        cl::Context::setDefault(context);
        _clContext = cl::Context::getDefault();
#else
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
#if 1
        cl::Device d = cl::Device::getDefault();
        std::cout << "OpenCL Default Device: " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "- Device Version: " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
        std::cout << "- Driver Version: " << d.getInfo<CL_DRIVER_VERSION>() << std::endl;
        std::cout << "- OpenCL C Version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
        std::cout << "- Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "- CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        std::cout << "- CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
#endif
#endif
//       TODO: FIGURE OUT WHY THIS IS COMMENTED IN DOUG's version
//        loadPrograms(programs);
    }

#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
        std::map<std::string, std::string>* getShadersMap() { return &cl_shaders; }
         std::map<std::string, std::vector<unsigned char>>* getBytecodeMap() { return &cl_bytecode; }
#endif


    // OCLContext(const std::string& shadersRootPath = "") : OCLContext({}, shadersRootPath) { }

    virtual ~OCLContext() {
        waitForCompletion();
    }

    cl::Context clContext() { return _clContext; }

    inline static std::vector<int> computeDivisors(const size_t val) {
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
    inline static cl::NDRange computeWorkGroupSizes(size_t width, size_t height) {
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
        std::cout << "computeWorkGroupSizes for " << width << ", " << height << ": "
                  << width_divisor << ", " << height_divisor
                  << " (" << width_divisor * height_divisor << ") of " <<  max_workgroup_size << std::endl;
        return cl::NDRange(width_divisor, height_divisor);
    }

    inline static cl::EnqueueArgs buildEnqueueArgs(size_t width, size_t height) {
        cl::NDRange global_workgroup_size = cl::NDRange(width, height);
        cl::NDRange local_workgroup_size = computeWorkGroupSizes(width, height);
        return cl::EnqueueArgs(global_workgroup_size, local_workgroup_size);
    }

    inline static cl::EnqueueArgs buildMaxEnqueueArgs(size_t width, size_t height) {
        cl::Device d = cl::Device::getDefault();
        const size_t max_workgroup_size = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        const int max_dimension = sqrtf(max_workgroup_size);

        cl::NDRange global_workgroup_size = cl::NDRange(roundTo(width, max_dimension), roundTo(height, max_dimension));
        cl::NDRange local_workgroup_size = computeWorkGroupSizes(max_dimension, max_dimension);
        return cl::EnqueueArgs(global_workgroup_size, local_workgroup_size);
    }

    std::string OpenCLSource(const std::string& shaderName, bool use_local_shaders = false) {
#if defined(__ANDROID__) && defined(USE_ASSET_MANAGER)
        if(use_local_shaders){
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
        }
        else {
            return cl_shaders[shaderName];
        }
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

    std::vector<unsigned char> OpenCLBinary(const std::string& shaderName) {
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

    void loadPrograms(const std::vector<std::string>& programNames, bool use_local_shaders = false) {
        cl::Program program;
        cl::Device device;
        try {
            device = cl::Device::getDefault();
#if (defined(USE_TEXT_SHADERS))


            std::vector<std::string> sources;
            for (const auto& p : programNames) {

                const auto& source = OpenCLSource(p + ".cl", use_local_shaders);
                __android_log_print(ANDROID_LOG_INFO, "foo",  "OpenCL Source:  %s", source.c_str());
                std::cout << "OpenCL Source: " << source << std::endl;
                sources.push_back(source);
            }

            program = cl::Program(sources);
#else
//            cl::Program::Binaries
            std::vector<std::vector<unsigned char>> binary_list = {};
            for (const auto& p : programNames) {
//                std::cout << "Opening: " << p << std::endl;
                std::vector<unsigned char> binary = OpenCLBinary(p + ".o");
                binary_list.push_back(binary);
            }
//            std::vector<unsigned char> binary_element;
//            std::vector<std::vector<unsigned char>> combined_binary_list(1, binary_element);
            std::vector<std::vector<unsigned char>> combined_binary_list;
//            int ctr = 0;
//            for(int i = 0; i < binary_list.size(); i++){
//                std::vector<unsigned char> b = binary_list[i];
//                __android_log_print(ANDROID_LOG_INFO, "OpenCL",  "Binary %d Size: %d", ctr, (int) b.size());
//                for(int j = 0; j < b.size(); j++){
//                    combined_binary_list[0].push_back(binary_list[i][j]);
//                }
//                ctr++;
////                break;
//            }

            std::vector<unsigned char> combined_binary;
//            combined_binary.
            for (const auto& binaryContent : binary_list) {
                combined_binary.insert(combined_binary.end(), binaryContent.begin(), binaryContent.end());
                __android_log_print(ANDROID_LOG_INFO, "OpenCL",  "Combined Binary Size: %d", (int) combined_binary.size());
            }
            combined_binary_list.push_back(combined_binary);
            __android_log_print(ANDROID_LOG_INFO, "OpenCL",  "Combined_binary_list_size: %d", (int) combined_binary.size());


            // cl::Program expects to get the same number of devices in this device list as there are binaries in the binary list
            std::vector<cl::Device> devices(combined_binary_list.size(), device);

            __android_log_print(ANDROID_LOG_INFO, "OpenCL",  "--------------Loading OpenCL Binaries---------");
            std::cout << "--------------Loading OpenCL Binaries---------" << std::endl;
//            program = cl::Program(_clContext, {device, device}, binary_list);
            std::vector< cl_int > error_codes(binary_list.size());
            program = cl::Program(_clContext, devices, combined_binary_list, &error_codes);
//            program = cl::Program(_clContext, {device}, {combined_binary}, &error_codes);
            __android_log_print(ANDROID_LOG_INFO, "OpenCL",  "--------------Loaded OpenCL Binaries---------");
            std::cout << "--------------Loaded OpenCL Binaries---------" << std::endl;
            for(int i = 0; i < error_codes.size(); i++) {
                __android_log_print(ANDROID_LOG_INFO, "foo",
                                    "OpenCL Program Error Codes: %s", clStatusToString(error_codes[i]).c_str());
            }

//            cl::Program program = cl::Program(_clContext, {device}, {OpenCLBinary(programNames[0] + ".o")});

#endif

            program.build(device, cl_options);

//            std::string name  = device.getInfo<CL_DEVICE_NAME>();
//            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
//            __android_log_print(ANDROID_LOG_INFO, "foo",  "Build log for: %s: %s", name.c_str(), buildlog.c_str());

            _program = program;
        } catch (const cl::BuildError& e) {
            std::cerr << "OpenCL Build Error - " << e.what() << ": " << clStatusToString(e.err()) << std::endl;
            // Print build info for all devices
            for (auto& pair : e.getBuildLog()) {
                std::cerr << pair.second << std::endl;
            }

            std::string name  = device.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
//            std::cerr << "Build log for " << name << ":" << std::endl
//                      << buildlog << std::endl;
            __android_log_print(ANDROID_LOG_INFO, "foo",  "Build log for: %s: %s", name.c_str(), buildlog.c_str());

            throw std::runtime_error("OpenCL Build Error");
        }
        catch (const cl::Error& e) {
            std::cerr << "OpenCL Error - " << e.what() << ": " << clStatusToString(e.err()) << std::endl;
            __android_log_print(ANDROID_LOG_INFO, "foo",  "OpenCL Error - %s: %s",
                                e.what(), clStatusToString(e.err()).c_str());

//            for (auto& pair : e.getBuildLog()) {
//                std::cerr << pair.second << std::endl;
//            }

//            std::string name  = device.getInfo<CL_DEVICE_NAME>();
//            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
////            std::cerr << "Build log for " << name << ":" << std::endl
////                      << buildlog << std::endl;
//            __android_log_print(ANDROID_LOG_INFO, "foo",  "Build log for: %s: %s", name.c_str(), buildlog.c_str());

            throw std::runtime_error("OpenCL Error");
        }
    }

    virtual void waitForCompletion() override {
        cl::finish();
//        __android_log_print(ANDROID_LOG_INFO, "OpenCL Debug",  "Error code: %d", errcode);
    }

    virtual platform_buffer* new_platform_buffer(size_t size, bool readOnly) override {
        return new ocl_buffer(_clContext, size, readOnly);
    }

    virtual platform_texture* new_platform_texture(int _width, int _height, texture::format format) override {
        return new ocl_texture(_clContext, _width, _height, format);
    }

    virtual void enqueue(const std::string& kernelName, const gls::size& gridSize, const gls::size& threadGroupSize,
                         std::function<void(GpuCommandEncoder*)> encodeKernelParameters,
                         std::function<void(void)> completionHandler) override {
        cl::Kernel kernel(_program, kernelName.c_str());
        OCLCommandEncoder encoder(kernel);

        encodeKernelParameters(&encoder);

        cl::CommandQueue queue = cl::CommandQueue::getDefault();

        cl::NDRange global_workgroup_size = cl::NDRange(gridSize.width, gridSize.height);
        cl::NDRange local_workgroup_size = cl::NDRange(threadGroupSize.width, threadGroupSize.height);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_workgroup_size, local_workgroup_size);
    }

    virtual void enqueue(const std::string& kernelName, const gls::size& gridSize,
                         std::function<void(GpuCommandEncoder*)> encodeKernelParameters,
                         std::function<void(void)> completionHandler) override {
        try {
            cl::Kernel kernel(_program, kernelName.c_str());
            OCLCommandEncoder encoder(kernel);

            encodeKernelParameters(&encoder);

            cl::CommandQueue queue = cl::CommandQueue::getDefault();

            cl::NDRange global_workgroup_size = cl::NDRange(gridSize.width, gridSize.height);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_workgroup_size);
        } catch (const cl::Error& e) {
            std::cerr << "OpenCL Kernel Error - " << kernelName << " - " << e.what() << ": " << clStatusToString(e.err()) << std::endl;
            __android_log_print(ANDROID_LOG_INFO, "foo",  "OpenCL Kernel Error %s - %s: %s",
                                kernelName.c_str(), e.what(), clStatusToString(e.err()).c_str());
            throw std::runtime_error("OpenCL Kernel Error");
        }
    }
};

}  // namespace gls

#endif /* gls_ocl_h */
