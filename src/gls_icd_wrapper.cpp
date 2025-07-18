/*******************************************************************************
 * Copyright (c) 2021-2022 Glass Imaging Inc.
 * Author: Fabio Riccardi <fabio@glass-imaging.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <dlfcn.h>

#include "gls_logging.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#define CL_TARGET_OPENCL_VERSION 220

#include "gls_icd_wrapper.h"

namespace CL_WRAPPER_NS {

static std::string TAG = "OpenCLWrapper";

static void bindOpenClAPIEntries(void* libopencl);

int bindOpenCLLibrary() {
    static bool opencl_loaded = false;

    if (opencl_loaded) {
        gls::logging::LogDebug(TAG) << "OpenCL already loaded" << std::endl;
        return 0;
    }

    static const char* kClLibName = "libOpenCL.so";

    void* libopencl = dlopen(kClLibName, RTLD_NOW | RTLD_LOCAL);
    if (libopencl) {
        bindOpenClAPIEntries(libopencl);
        opencl_loaded = true;
        return 0;
    }

    std::string error(dlerror());
    gls::logging::LogError(TAG) << "Can not open OpenCL library on this device - " << error << std::endl;
    return -1;
}

#define bindEntry(name)                                                                           \
    {                                                                                             \
        name = reinterpret_cast<cl_api_##name>(dlsym(libopencl, #name));                          \
        if (!name) {                                                                              \
            gls::logging::LogDebug(TAG) << "Couldn't bind OpenCL API entry: " #name << std::endl; \
        }                                                                                         \
    }

// Bind up to OpenCL 2.0 by default

static void bindOpenClAPIEntries(void* libopencl) {
    /* OpenCL 1.0 */
    bindEntry(clGetPlatformIDs);
    bindEntry(clGetPlatformInfo);
    bindEntry(clGetDeviceIDs);
    bindEntry(clGetDeviceInfo);
    bindEntry(clCreateContext);
    bindEntry(clCreateContextFromType);
    bindEntry(clRetainContext);
    bindEntry(clReleaseContext);
    bindEntry(clGetContextInfo);
    bindEntry(clCreateCommandQueue);
    bindEntry(clRetainCommandQueue);
    bindEntry(clReleaseCommandQueue);
    bindEntry(clGetCommandQueueInfo);
    bindEntry(clSetCommandQueueProperty);
    bindEntry(clCreateBuffer);
    bindEntry(clCreateImage2D);
    bindEntry(clCreateImage3D);
    bindEntry(clRetainMemObject);
    bindEntry(clReleaseMemObject);
    bindEntry(clGetSupportedImageFormats);
    bindEntry(clGetMemObjectInfo);
    bindEntry(clGetImageInfo);
    bindEntry(clCreateSampler);
    bindEntry(clRetainSampler);
    bindEntry(clReleaseSampler);
    bindEntry(clGetSamplerInfo);
    bindEntry(clCreateProgramWithSource);
    bindEntry(clCreateProgramWithBinary);
    bindEntry(clRetainProgram);
    bindEntry(clReleaseProgram);
    bindEntry(clBuildProgram);
    bindEntry(clUnloadCompiler);
    bindEntry(clGetProgramInfo);
    bindEntry(clGetProgramBuildInfo);
    bindEntry(clCreateKernel);
    bindEntry(clCreateKernelsInProgram);
    bindEntry(clRetainKernel);
    bindEntry(clReleaseKernel);
    bindEntry(clSetKernelArg);
    bindEntry(clGetKernelInfo);
    bindEntry(clGetKernelWorkGroupInfo);
    bindEntry(clWaitForEvents);
    bindEntry(clGetEventInfo);
    bindEntry(clRetainEvent);
    bindEntry(clReleaseEvent);
    bindEntry(clGetEventProfilingInfo);
    bindEntry(clFlush);
    bindEntry(clFinish);
    bindEntry(clEnqueueReadBuffer);
    bindEntry(clEnqueueWriteBuffer);
    bindEntry(clEnqueueCopyBuffer);
    bindEntry(clEnqueueReadImage);
    bindEntry(clEnqueueWriteImage);
    bindEntry(clEnqueueCopyImage);
    bindEntry(clEnqueueCopyImageToBuffer);
    bindEntry(clEnqueueCopyBufferToImage);
    bindEntry(clEnqueueMapBuffer);
    bindEntry(clEnqueueMapImage);
    bindEntry(clEnqueueUnmapMemObject);
    bindEntry(clEnqueueNDRangeKernel);
    bindEntry(clEnqueueTask);
    bindEntry(clEnqueueNativeKernel);
    bindEntry(clEnqueueMarker);
    bindEntry(clEnqueueWaitForEvents);
    bindEntry(clEnqueueBarrier);
    bindEntry(clGetExtensionFunctionAddress);
    bindEntry(clCreateFromGLBuffer);
    bindEntry(clCreateFromGLTexture2D);
    bindEntry(clCreateFromGLTexture3D);
    bindEntry(clCreateFromGLRenderbuffer);
    bindEntry(clGetGLObjectInfo);
    bindEntry(clGetGLTextureInfo);
    bindEntry(clEnqueueAcquireGLObjects);
    bindEntry(clEnqueueReleaseGLObjects);
    bindEntry(clGetGLContextInfoKHR);

    /* OpenCL 1.1 */
    bindEntry(clSetEventCallback);
    bindEntry(clCreateSubBuffer);
    bindEntry(clSetMemObjectDestructorCallback);
    bindEntry(clCreateUserEvent);
    bindEntry(clSetUserEventStatus);
    bindEntry(clEnqueueReadBufferRect);
    bindEntry(clEnqueueWriteBufferRect);
    bindEntry(clEnqueueCopyBufferRect);

    /* cl_ext_device_fission */
    bindEntry(clCreateSubDevicesEXT);
    bindEntry(clRetainDeviceEXT);
    bindEntry(clReleaseDeviceEXT);

    /* cl_khr_gl_event */
    bindEntry(clCreateEventFromGLsyncKHR);

    /* OpenCL 1.2 */
    bindEntry(clCreateSubDevices);
    bindEntry(clRetainDevice);
    bindEntry(clReleaseDevice);
    bindEntry(clCreateImage);
    bindEntry(clCreateProgramWithBuiltInKernels);
    bindEntry(clCompileProgram);
    bindEntry(clLinkProgram);
    bindEntry(clUnloadPlatformCompiler);
    bindEntry(clGetKernelArgInfo);
    bindEntry(clEnqueueFillBuffer);
    bindEntry(clEnqueueFillImage);
    bindEntry(clEnqueueMigrateMemObjects);
    bindEntry(clEnqueueMarkerWithWaitList);
    bindEntry(clEnqueueBarrierWithWaitList);
    bindEntry(clGetExtensionFunctionAddressForPlatform);
    bindEntry(clCreateFromGLTexture);

#ifdef _WIN32
    /* cl_khr_d3d10_sharing */
    bindEntry(clGetDeviceIDsFromD3D10KHR);
    bindEntry(clCreateFromD3D10BufferKHR);
    bindEntry(clCreateFromD3D10Texture2DKHR);
    bindEntry(clCreateFromD3D10Texture3DKHR);
    bindEntry(clEnqueueAcquireD3D10ObjectsKHR);
    bindEntry(clEnqueueReleaseD3D10ObjectsKHR);

    /* cl_khr_d3d11_sharing */
    bindEntry(clGetDeviceIDsFromD3D11KHR);
    bindEntry(clCreateFromD3D11BufferKHR);
    bindEntry(clCreateFromD3D11Texture2DKHR);
    bindEntry(clCreateFromD3D11Texture3DKHR);
    bindEntry(clCreateFromDX9MediaSurfaceKHR);
    bindEntry(clEnqueueAcquireD3D11ObjectsKHR);
    bindEntry(clEnqueueReleaseD3D11ObjectsKHR);

    /* cl_khr_dx9_media_sharing */
    bindEntry(clGetDeviceIDsFromDX9MediaAdapterKHR);
    bindEntry(clEnqueueAcquireDX9MediaSurfacesKHR);
    bindEntry(clEnqueueReleaseDX9MediaSurfacesKHR);
#endif

    /* cl_khr_egl_image */
    bindEntry(clCreateFromEGLImageKHR);
    bindEntry(clEnqueueAcquireEGLObjectsKHR);
    bindEntry(clEnqueueReleaseEGLObjectsKHR);

    /* cl_khr_egl_event */
    bindEntry(clCreateEventFromEGLSyncKHR);

    /* OpenCL 2.0 */
    bindEntry(clCreateCommandQueueWithProperties);
    bindEntry(clCreatePipe);
    bindEntry(clGetPipeInfo);
    bindEntry(clSVMAlloc);
    bindEntry(clSVMFree);
    bindEntry(clEnqueueSVMFree);
    bindEntry(clEnqueueSVMMemcpy);
    bindEntry(clEnqueueSVMMemFill);
    bindEntry(clEnqueueSVMMap);
    bindEntry(clEnqueueSVMUnmap);
    bindEntry(clCreateSamplerWithProperties);
    bindEntry(clSetKernelArgSVMPointer);
    bindEntry(clSetKernelExecInfo);

    /* cl_khr_sub_groups */
    bindEntry(clGetKernelSubGroupInfoKHR);

    /* cl_qcom_recordable_queues extension */
    bindEntry(clNewRecordingQCOM);
    bindEntry(clEndRecordingQCOM);
    bindEntry(clReleaseRecordingQCOM);
    bindEntry(clRetainRecordingQCOM);
    bindEntry(clEnqueueRecordingQCOM);

#ifdef OPENCL_2_1
    /* OpenCL 2.1 */
    bindEntry(clCloneKernel);
    bindEntry(clCreateProgramWithIL);
    bindEntry(clEnqueueSVMMigrateMem);
    bindEntry(clGetDeviceAndHostTimer);
    bindEntry(clGetHostTimer);
    bindEntry(clGetKernelSubGroupInfo);
    bindEntry(clSetDefaultDeviceCommandQueue);
#endif

#ifdef OPENCL_2_1
    /* OpenCL 2.2 */
    bindEntry(clSetProgramReleaseCallback);
    bindEntry(clSetProgramSpecializationConstant);
#endif

#ifdef OPENCL_3_0
    /* OpenCL 3.0 */
    bindEntry(clCreateBufferWithProperties);
    bindEntry(clCreateImageWithProperties);
    bindEntry(clSetContextDestructorCallback);
#endif
}

/* OpenCL 1.0 */
cl_api_clGetPlatformIDs clGetPlatformIDs = nullptr;
cl_api_clGetPlatformInfo clGetPlatformInfo = nullptr;
cl_api_clGetDeviceIDs clGetDeviceIDs = nullptr;
cl_api_clGetDeviceInfo clGetDeviceInfo = nullptr;
cl_api_clCreateContext clCreateContext = nullptr;
cl_api_clCreateContextFromType clCreateContextFromType = nullptr;
cl_api_clRetainContext clRetainContext = nullptr;
cl_api_clReleaseContext clReleaseContext = nullptr;
cl_api_clGetContextInfo clGetContextInfo = nullptr;
cl_api_clCreateCommandQueue clCreateCommandQueue = nullptr;
cl_api_clRetainCommandQueue clRetainCommandQueue = nullptr;
cl_api_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
cl_api_clGetCommandQueueInfo clGetCommandQueueInfo = nullptr;
cl_api_clSetCommandQueueProperty clSetCommandQueueProperty = nullptr;
cl_api_clCreateBuffer clCreateBuffer = nullptr;
cl_api_clCreateImage2D clCreateImage2D = nullptr;
cl_api_clCreateImage3D clCreateImage3D = nullptr;
cl_api_clRetainMemObject clRetainMemObject = nullptr;
cl_api_clReleaseMemObject clReleaseMemObject = nullptr;
cl_api_clGetSupportedImageFormats clGetSupportedImageFormats = nullptr;
cl_api_clGetMemObjectInfo clGetMemObjectInfo = nullptr;
cl_api_clGetImageInfo clGetImageInfo = nullptr;
cl_api_clCreateSampler clCreateSampler = nullptr;
cl_api_clRetainSampler clRetainSampler = nullptr;
cl_api_clReleaseSampler clReleaseSampler = nullptr;
cl_api_clGetSamplerInfo clGetSamplerInfo = nullptr;
cl_api_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
cl_api_clCreateProgramWithBinary clCreateProgramWithBinary = nullptr;
cl_api_clRetainProgram clRetainProgram = nullptr;
cl_api_clReleaseProgram clReleaseProgram = nullptr;
cl_api_clBuildProgram clBuildProgram = nullptr;
cl_api_clUnloadCompiler clUnloadCompiler = nullptr;
cl_api_clGetProgramInfo clGetProgramInfo = nullptr;
cl_api_clGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
cl_api_clCreateKernel clCreateKernel = nullptr;
cl_api_clCreateKernelsInProgram clCreateKernelsInProgram = nullptr;
cl_api_clRetainKernel clRetainKernel = nullptr;
cl_api_clReleaseKernel clReleaseKernel = nullptr;
cl_api_clSetKernelArg clSetKernelArg = nullptr;
cl_api_clGetKernelInfo clGetKernelInfo = nullptr;
cl_api_clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo = nullptr;
cl_api_clWaitForEvents clWaitForEvents = nullptr;
cl_api_clGetEventInfo clGetEventInfo = nullptr;
cl_api_clRetainEvent clRetainEvent = nullptr;
cl_api_clReleaseEvent clReleaseEvent = nullptr;
cl_api_clGetEventProfilingInfo clGetEventProfilingInfo = nullptr;
cl_api_clFlush clFlush = nullptr;
cl_api_clFinish clFinish = nullptr;
cl_api_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
cl_api_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
cl_api_clEnqueueCopyBuffer clEnqueueCopyBuffer = nullptr;
cl_api_clEnqueueReadImage clEnqueueReadImage = nullptr;
cl_api_clEnqueueWriteImage clEnqueueWriteImage = nullptr;
cl_api_clEnqueueCopyImage clEnqueueCopyImage = nullptr;
cl_api_clEnqueueCopyImageToBuffer clEnqueueCopyImageToBuffer = nullptr;
cl_api_clEnqueueCopyBufferToImage clEnqueueCopyBufferToImage = nullptr;
cl_api_clEnqueueMapBuffer clEnqueueMapBuffer = nullptr;
cl_api_clEnqueueMapImage clEnqueueMapImage = nullptr;
cl_api_clEnqueueUnmapMemObject clEnqueueUnmapMemObject = nullptr;
cl_api_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
cl_api_clEnqueueTask clEnqueueTask = nullptr;
cl_api_clEnqueueNativeKernel clEnqueueNativeKernel = nullptr;
cl_api_clEnqueueMarker clEnqueueMarker = nullptr;
cl_api_clEnqueueWaitForEvents clEnqueueWaitForEvents = nullptr;
cl_api_clEnqueueBarrier clEnqueueBarrier = nullptr;
cl_api_clGetExtensionFunctionAddress clGetExtensionFunctionAddress = nullptr;
cl_api_clCreateFromGLBuffer clCreateFromGLBuffer = nullptr;
cl_api_clCreateFromGLTexture2D clCreateFromGLTexture2D = nullptr;
cl_api_clCreateFromGLTexture3D clCreateFromGLTexture3D = nullptr;
cl_api_clCreateFromGLRenderbuffer clCreateFromGLRenderbuffer = nullptr;
cl_api_clGetGLObjectInfo clGetGLObjectInfo = nullptr;
cl_api_clGetGLTextureInfo clGetGLTextureInfo = nullptr;
cl_api_clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects = nullptr;
cl_api_clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects = nullptr;
cl_api_clGetGLContextInfoKHR clGetGLContextInfoKHR = nullptr;

/* cl_khr_d3d10_sharing */
cl_api_clGetDeviceIDsFromD3D10KHR clGetDeviceIDsFromD3D10KHR = nullptr;
cl_api_clCreateFromD3D10BufferKHR clCreateFromD3D10BufferKHR = nullptr;
cl_api_clCreateFromD3D10Texture2DKHR clCreateFromD3D10Texture2DKHR = nullptr;
cl_api_clCreateFromD3D10Texture3DKHR clCreateFromD3D10Texture3DKHR = nullptr;
cl_api_clEnqueueAcquireD3D10ObjectsKHR clEnqueueAcquireD3D10ObjectsKHR = nullptr;
cl_api_clEnqueueReleaseD3D10ObjectsKHR clEnqueueReleaseD3D10ObjectsKHR = nullptr;

/* OpenCL 1.1 */
cl_api_clSetEventCallback clSetEventCallback = nullptr;
cl_api_clCreateSubBuffer clCreateSubBuffer = nullptr;
cl_api_clSetMemObjectDestructorCallback clSetMemObjectDestructorCallback = nullptr;
cl_api_clCreateUserEvent clCreateUserEvent = nullptr;
cl_api_clSetUserEventStatus clSetUserEventStatus = nullptr;
cl_api_clEnqueueReadBufferRect clEnqueueReadBufferRect = nullptr;
cl_api_clEnqueueWriteBufferRect clEnqueueWriteBufferRect = nullptr;
cl_api_clEnqueueCopyBufferRect clEnqueueCopyBufferRect = nullptr;

/* cl_ext_device_fission */
cl_api_clCreateSubDevicesEXT clCreateSubDevicesEXT = nullptr;
cl_api_clRetainDeviceEXT clRetainDeviceEXT = nullptr;
cl_api_clReleaseDeviceEXT clReleaseDeviceEXT = nullptr;

/* cl_khr_gl_event */
cl_api_clCreateEventFromGLsyncKHR clCreateEventFromGLsyncKHR = nullptr;

/* OpenCL 1.2 */
cl_api_clCreateSubDevices clCreateSubDevices = nullptr;
cl_api_clRetainDevice clRetainDevice = nullptr;
cl_api_clReleaseDevice clReleaseDevice = nullptr;
cl_api_clCreateImage clCreateImage = nullptr;
cl_api_clCreateProgramWithBuiltInKernels clCreateProgramWithBuiltInKernels = nullptr;
cl_api_clCompileProgram clCompileProgram = nullptr;
cl_api_clLinkProgram clLinkProgram = nullptr;
cl_api_clUnloadPlatformCompiler clUnloadPlatformCompiler = nullptr;
cl_api_clGetKernelArgInfo clGetKernelArgInfo = nullptr;
cl_api_clEnqueueFillBuffer clEnqueueFillBuffer = nullptr;
cl_api_clEnqueueFillImage clEnqueueFillImage = nullptr;
cl_api_clEnqueueMigrateMemObjects clEnqueueMigrateMemObjects = nullptr;
cl_api_clEnqueueMarkerWithWaitList clEnqueueMarkerWithWaitList = nullptr;
cl_api_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList = nullptr;
cl_api_clGetExtensionFunctionAddressForPlatform clGetExtensionFunctionAddressForPlatform = nullptr;
cl_api_clCreateFromGLTexture clCreateFromGLTexture = nullptr;

/* cl_khr_d3d11_sharing */
cl_api_clGetDeviceIDsFromD3D11KHR clGetDeviceIDsFromD3D11KHR = nullptr;
cl_api_clCreateFromD3D11BufferKHR clCreateFromD3D11BufferKHR = nullptr;
cl_api_clCreateFromD3D11Texture2DKHR clCreateFromD3D11Texture2DKHR = nullptr;
cl_api_clCreateFromD3D11Texture3DKHR clCreateFromD3D11Texture3DKHR = nullptr;
cl_api_clCreateFromDX9MediaSurfaceKHR clCreateFromDX9MediaSurfaceKHR = nullptr;
cl_api_clEnqueueAcquireD3D11ObjectsKHR clEnqueueAcquireD3D11ObjectsKHR = nullptr;
cl_api_clEnqueueReleaseD3D11ObjectsKHR clEnqueueReleaseD3D11ObjectsKHR = nullptr;

/* cl_qcom_recordable_queues extension */
cl_api_clNewRecordingQCOM clNewRecordingQCOM = nullptr;
cl_api_clEndRecordingQCOM clEndRecordingQCOM = nullptr;
cl_api_clReleaseRecordingQCOM clReleaseRecordingQCOM = nullptr;
cl_api_clRetainRecordingQCOM clRetainRecordingQCOM = nullptr;
cl_api_clEnqueueRecordingQCOM clEnqueueRecordingQCOM = nullptr;

/* cl_khr_dx9_media_sharing */
cl_api_clGetDeviceIDsFromDX9MediaAdapterKHR clGetDeviceIDsFromDX9MediaAdapterKHR = nullptr;
cl_api_clEnqueueAcquireDX9MediaSurfacesKHR clEnqueueAcquireDX9MediaSurfacesKHR = nullptr;
cl_api_clEnqueueReleaseDX9MediaSurfacesKHR clEnqueueReleaseDX9MediaSurfacesKHR = nullptr;

/* cl_khr_egl_image */
cl_api_clCreateFromEGLImageKHR clCreateFromEGLImageKHR = nullptr;
cl_api_clEnqueueAcquireEGLObjectsKHR clEnqueueAcquireEGLObjectsKHR = nullptr;
cl_api_clEnqueueReleaseEGLObjectsKHR clEnqueueReleaseEGLObjectsKHR = nullptr;

/* cl_khr_egl_event */
cl_api_clCreateEventFromEGLSyncKHR clCreateEventFromEGLSyncKHR = nullptr;

/* OpenCL 2.0 */
cl_api_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties = nullptr;
cl_api_clCreatePipe clCreatePipe = nullptr;
cl_api_clGetPipeInfo clGetPipeInfo = nullptr;
cl_api_clSVMAlloc clSVMAlloc = nullptr;
cl_api_clSVMFree clSVMFree = nullptr;
cl_api_clEnqueueSVMFree clEnqueueSVMFree = nullptr;
cl_api_clEnqueueSVMMemcpy clEnqueueSVMMemcpy = nullptr;
cl_api_clEnqueueSVMMemFill clEnqueueSVMMemFill = nullptr;
cl_api_clEnqueueSVMMap clEnqueueSVMMap = nullptr;
cl_api_clEnqueueSVMUnmap clEnqueueSVMUnmap = nullptr;
cl_api_clCreateSamplerWithProperties clCreateSamplerWithProperties = nullptr;
cl_api_clSetKernelArgSVMPointer clSetKernelArgSVMPointer = nullptr;
cl_api_clSetKernelExecInfo clSetKernelExecInfo = nullptr;

/* cl_khr_sub_groups */
cl_api_clGetKernelSubGroupInfoKHR clGetKernelSubGroupInfoKHR = nullptr;

/* OpenCL 2.1 */
cl_api_clCloneKernel clCloneKernel = nullptr;
cl_api_clCreateProgramWithIL clCreateProgramWithIL = nullptr;
cl_api_clEnqueueSVMMigrateMem clEnqueueSVMMigrateMem = nullptr;
cl_api_clGetDeviceAndHostTimer clGetDeviceAndHostTimer = nullptr;
cl_api_clGetHostTimer clGetHostTimer = nullptr;
cl_api_clGetKernelSubGroupInfo clGetKernelSubGroupInfo = nullptr;
cl_api_clSetDefaultDeviceCommandQueue clSetDefaultDeviceCommandQueue = nullptr;

/* OpenCL 2.2 */
cl_api_clSetProgramReleaseCallback clSetProgramReleaseCallback = nullptr;
cl_api_clSetProgramSpecializationConstant clSetProgramSpecializationConstant = nullptr;

/* OpenCL 3.0 */
cl_api_clCreateBufferWithProperties clCreateBufferWithProperties = nullptr;
cl_api_clCreateImageWithProperties clCreateImageWithProperties = nullptr;
cl_api_clSetContextDestructorCallback clSetContextDestructorCallback = nullptr;
}  // namespace CL_WRAPPER_NS

#pragma clang diagnostic pop
