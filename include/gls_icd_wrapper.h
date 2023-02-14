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

#ifndef GLS_ICD_WRAPPER_H
#define GLS_ICD_WRAPPER_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "CL/cl_icd.h"

#define CL_WRAPPER_NS opencl
#define CL_USE_FUNCTION_POINTERS

namespace CL_WRAPPER_NS {

int bindOpenCLLibrary();

/* OpenCL 1.0 */
extern cl_api_clGetPlatformIDs clGetPlatformIDs;
extern cl_api_clGetPlatformInfo clGetPlatformInfo;
extern cl_api_clGetDeviceIDs clGetDeviceIDs;
extern cl_api_clGetDeviceInfo clGetDeviceInfo;
extern cl_api_clCreateContext clCreateContext;
extern cl_api_clCreateContextFromType clCreateContextFromType;
extern cl_api_clRetainContext clRetainContext;
extern cl_api_clReleaseContext clReleaseContext;
extern cl_api_clGetContextInfo clGetContextInfo;
extern cl_api_clCreateCommandQueue clCreateCommandQueue;
extern cl_api_clRetainCommandQueue clRetainCommandQueue;
extern cl_api_clReleaseCommandQueue clReleaseCommandQueue;
extern cl_api_clGetCommandQueueInfo clGetCommandQueueInfo;
extern cl_api_clSetCommandQueueProperty clSetCommandQueueProperty;
extern cl_api_clCreateBuffer clCreateBuffer;
extern cl_api_clCreateImage2D clCreateImage2D;
extern cl_api_clCreateImage3D clCreateImage3D;
extern cl_api_clRetainMemObject clRetainMemObject;
extern cl_api_clReleaseMemObject clReleaseMemObject;
extern cl_api_clGetSupportedImageFormats clGetSupportedImageFormats;
extern cl_api_clGetMemObjectInfo clGetMemObjectInfo;
extern cl_api_clGetImageInfo clGetImageInfo;
extern cl_api_clCreateSampler clCreateSampler;
extern cl_api_clRetainSampler clRetainSampler;
extern cl_api_clReleaseSampler clReleaseSampler;
extern cl_api_clGetSamplerInfo clGetSamplerInfo;
extern cl_api_clCreateProgramWithSource clCreateProgramWithSource;
extern cl_api_clCreateProgramWithBinary clCreateProgramWithBinary;
extern cl_api_clRetainProgram clRetainProgram;
extern cl_api_clReleaseProgram clReleaseProgram;
extern cl_api_clBuildProgram clBuildProgram;
extern cl_api_clUnloadCompiler clUnloadCompiler;
extern cl_api_clGetProgramInfo clGetProgramInfo;
extern cl_api_clGetProgramBuildInfo clGetProgramBuildInfo;
extern cl_api_clCreateKernel clCreateKernel;
extern cl_api_clCreateKernelsInProgram clCreateKernelsInProgram;
extern cl_api_clRetainKernel clRetainKernel;
extern cl_api_clReleaseKernel clReleaseKernel;
extern cl_api_clSetKernelArg clSetKernelArg;
extern cl_api_clGetKernelInfo clGetKernelInfo;
extern cl_api_clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo;
extern cl_api_clWaitForEvents clWaitForEvents;
extern cl_api_clGetEventInfo clGetEventInfo;
extern cl_api_clRetainEvent clRetainEvent;
extern cl_api_clReleaseEvent clReleaseEvent;
extern cl_api_clGetEventProfilingInfo clGetEventProfilingInfo;
extern cl_api_clFlush clFlush;
extern cl_api_clFinish clFinish;
extern cl_api_clEnqueueReadBuffer clEnqueueReadBuffer;
extern cl_api_clEnqueueWriteBuffer clEnqueueWriteBuffer;
extern cl_api_clEnqueueCopyBuffer clEnqueueCopyBuffer;
extern cl_api_clEnqueueReadImage clEnqueueReadImage;
extern cl_api_clEnqueueWriteImage clEnqueueWriteImage;
extern cl_api_clEnqueueCopyImage clEnqueueCopyImage;
extern cl_api_clEnqueueCopyImageToBuffer clEnqueueCopyImageToBuffer;
extern cl_api_clEnqueueCopyBufferToImage clEnqueueCopyBufferToImage;
extern cl_api_clEnqueueMapBuffer clEnqueueMapBuffer;
extern cl_api_clEnqueueMapImage clEnqueueMapImage;
extern cl_api_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
extern cl_api_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
extern cl_api_clEnqueueTask clEnqueueTask;
extern cl_api_clEnqueueNativeKernel clEnqueueNativeKernel;
extern cl_api_clEnqueueMarker clEnqueueMarker;
extern cl_api_clEnqueueWaitForEvents clEnqueueWaitForEvents;
extern cl_api_clEnqueueBarrier clEnqueueBarrier;
extern cl_api_clGetExtensionFunctionAddress clGetExtensionFunctionAddress;
extern cl_api_clCreateFromGLBuffer clCreateFromGLBuffer;
extern cl_api_clCreateFromGLTexture2D clCreateFromGLTexture2D;
extern cl_api_clCreateFromGLTexture3D clCreateFromGLTexture3D;
extern cl_api_clCreateFromGLRenderbuffer clCreateFromGLRenderbuffer;
extern cl_api_clGetGLObjectInfo clGetGLObjectInfo;
extern cl_api_clGetGLTextureInfo clGetGLTextureInfo;
extern cl_api_clEnqueueAcquireGLObjects clEnqueueAcquireGLObjects;
extern cl_api_clEnqueueReleaseGLObjects clEnqueueReleaseGLObjects;
extern cl_api_clGetGLContextInfoKHR clGetGLContextInfoKHR;

/* cl_khr_d3d10_sharing */
extern cl_api_clGetDeviceIDsFromD3D10KHR clGetDeviceIDsFromD3D10KHR;
extern cl_api_clCreateFromD3D10BufferKHR clCreateFromD3D10BufferKHR;
extern cl_api_clCreateFromD3D10Texture2DKHR clCreateFromD3D10Texture2DKHR;
extern cl_api_clCreateFromD3D10Texture3DKHR clCreateFromD3D10Texture3DKHR;
extern cl_api_clEnqueueAcquireD3D10ObjectsKHR clEnqueueAcquireD3D10ObjectsKHR;
extern cl_api_clEnqueueReleaseD3D10ObjectsKHR clEnqueueReleaseD3D10ObjectsKHR;

/* OpenCL 1.1 */
extern cl_api_clSetEventCallback clSetEventCallback;
extern cl_api_clCreateSubBuffer clCreateSubBuffer;
extern cl_api_clSetMemObjectDestructorCallback clSetMemObjectDestructorCallback;
extern cl_api_clCreateUserEvent clCreateUserEvent;
extern cl_api_clSetUserEventStatus clSetUserEventStatus;
extern cl_api_clEnqueueReadBufferRect clEnqueueReadBufferRect;
extern cl_api_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
extern cl_api_clEnqueueCopyBufferRect clEnqueueCopyBufferRect;

/* cl_ext_device_fission */
extern cl_api_clCreateSubDevicesEXT clCreateSubDevicesEXT;
extern cl_api_clRetainDeviceEXT clRetainDeviceEXT;
extern cl_api_clReleaseDeviceEXT clReleaseDeviceEXT;

/* cl_khr_gl_event */
extern cl_api_clCreateEventFromGLsyncKHR clCreateEventFromGLsyncKHR;

/* OpenCL 1.2 */
extern cl_api_clCreateSubDevices clCreateSubDevices;
extern cl_api_clRetainDevice clRetainDevice;
extern cl_api_clReleaseDevice clReleaseDevice;
extern cl_api_clCreateImage clCreateImage;
extern cl_api_clCreateProgramWithBuiltInKernels clCreateProgramWithBuiltInKernels;
extern cl_api_clCompileProgram clCompileProgram;
extern cl_api_clLinkProgram clLinkProgram;
extern cl_api_clUnloadPlatformCompiler clUnloadPlatformCompiler;
extern cl_api_clGetKernelArgInfo clGetKernelArgInfo;
extern cl_api_clEnqueueFillBuffer clEnqueueFillBuffer;
extern cl_api_clEnqueueFillImage clEnqueueFillImage;
extern cl_api_clEnqueueMigrateMemObjects clEnqueueMigrateMemObjects;
extern cl_api_clEnqueueMarkerWithWaitList clEnqueueMarkerWithWaitList;
extern cl_api_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;
extern cl_api_clGetExtensionFunctionAddressForPlatform
        clGetExtensionFunctionAddressForPlatform;
extern cl_api_clCreateFromGLTexture clCreateFromGLTexture;

/* cl_khr_d3d11_sharing */
extern cl_api_clGetDeviceIDsFromD3D11KHR clGetDeviceIDsFromD3D11KHR;
extern cl_api_clCreateFromD3D11BufferKHR clCreateFromD3D11BufferKHR;
extern cl_api_clCreateFromD3D11Texture2DKHR clCreateFromD3D11Texture2DKHR;
extern cl_api_clCreateFromD3D11Texture3DKHR clCreateFromD3D11Texture3DKHR;
extern cl_api_clCreateFromDX9MediaSurfaceKHR clCreateFromDX9MediaSurfaceKHR;
extern cl_api_clEnqueueAcquireD3D11ObjectsKHR clEnqueueAcquireD3D11ObjectsKHR;
extern cl_api_clEnqueueReleaseD3D11ObjectsKHR clEnqueueReleaseD3D11ObjectsKHR;

/* cl_khr_dx9_media_sharing */
extern cl_api_clGetDeviceIDsFromDX9MediaAdapterKHR
        clGetDeviceIDsFromDX9MediaAdapterKHR;
extern cl_api_clEnqueueAcquireDX9MediaSurfacesKHR
        clEnqueueAcquireDX9MediaSurfacesKHR;
extern cl_api_clEnqueueReleaseDX9MediaSurfacesKHR
        clEnqueueReleaseDX9MediaSurfacesKHR;

/* cl_khr_egl_image */
extern cl_api_clCreateFromEGLImageKHR clCreateFromEGLImageKHR;
extern cl_api_clEnqueueAcquireEGLObjectsKHR clEnqueueAcquireEGLObjectsKHR;
extern cl_api_clEnqueueReleaseEGLObjectsKHR clEnqueueReleaseEGLObjectsKHR;

/* cl_khr_egl_event */
extern cl_api_clCreateEventFromEGLSyncKHR clCreateEventFromEGLSyncKHR;

/* OpenCL 2.0 */
extern cl_api_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties;
extern cl_api_clCreatePipe clCreatePipe;
extern cl_api_clGetPipeInfo clGetPipeInfo;
extern cl_api_clSVMAlloc clSVMAlloc;
extern cl_api_clSVMFree clSVMFree;
extern cl_api_clEnqueueSVMFree clEnqueueSVMFree;
extern cl_api_clEnqueueSVMMemcpy clEnqueueSVMMemcpy;
extern cl_api_clEnqueueSVMMemFill clEnqueueSVMMemFill;
extern cl_api_clEnqueueSVMMap clEnqueueSVMMap;
extern cl_api_clEnqueueSVMUnmap clEnqueueSVMUnmap;
extern cl_api_clCreateSamplerWithProperties clCreateSamplerWithProperties;
extern cl_api_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
extern cl_api_clSetKernelExecInfo clSetKernelExecInfo;

/* cl_khr_sub_groups */
extern cl_api_clGetKernelSubGroupInfoKHR clGetKernelSubGroupInfoKHR;

/* OpenCL 2.1 */
extern cl_api_clCloneKernel clCloneKernel;
extern cl_api_clCreateProgramWithIL clCreateProgramWithIL;
extern cl_api_clEnqueueSVMMigrateMem clEnqueueSVMMigrateMem;
extern cl_api_clGetDeviceAndHostTimer clGetDeviceAndHostTimer;
extern cl_api_clGetHostTimer clGetHostTimer;
extern cl_api_clGetKernelSubGroupInfo clGetKernelSubGroupInfo;
extern cl_api_clSetDefaultDeviceCommandQueue clSetDefaultDeviceCommandQueue;

/* OpenCL 2.2 */
extern cl_api_clSetProgramReleaseCallback clSetProgramReleaseCallback;
extern cl_api_clSetProgramSpecializationConstant clSetProgramSpecializationConstant;

/* OpenCL 3.0 */
extern cl_api_clCreateBufferWithProperties clCreateBufferWithProperties;
extern cl_api_clCreateImageWithProperties clCreateImageWithProperties;
extern cl_api_clSetContextDestructorCallback clSetContextDestructorCallback;

#pragma clang diagnostic pop

}
#endif  // GLS_ICD_WRAPPER_H
