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

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#else
#error "OPENCL EXTENSION cl_khr_fp16 unavailable"
#endif

#define TEXTURE(id)
#define BUFFER(id)

#define USE_GLOBAL_ID

#define VAL(type)       const type

#define __ovld __attribute__((overloadable))

#define image2dh_t image2d_t
#define image2df_t image2d_t
#define image2dui_t image2d_t

#define image2dh_write_t write_only image2d_t
#define image2df_write_t write_only image2d_t
#define image2dui_write_t write_only image2d_t

#define uint2(args...)      ((uint2) (args))
#define uint3(args...)      ((uint3) (args))
#define uint4(args...)      ((uint4) (args))
#define uint8(args...)      ((uint8) (args))

#define int2(args...)       ((int2) (args))
#define int3(args...)       ((int3) (args))
#define int4(args...)       ((int4) (args))
#define int8(args...)       ((int8) (args))

#define half2(args...)      ((half2) (args))
#define half3(args...)      ((half3) (args))
#define half4(args...)      ((half4) (args))
#define half8(args...)      ((half8) (args))

#define float2(args...)     ((float2) (args))
#define float3(args...)     ((float3) (args))
#define float4(args...)     ((float4) (args))
#define float8(args...)     ((float8) (args))

//#ifdef __APPLE__
//static inline __ovld half2 myconvert_half2(float2 val) {
//    return (half2) (val.x, val.y);
//}
//
//static inline __ovld half3 myconvert_half3(float3 val) {
//    return (half3) (val.x, val.y, val.z);
//}
//
//static inline __ovld half4 myconvert_half4(float4 val) {
//    return (half4) (val.x, val.y, val.z, val.w);
//}
//
//#define convert_half2(val)      myconvert_half2(val)
//#define convert_half3(val)      myconvert_half3(val)
//#define convert_half4(val)      myconvert_half4(val)
//
//#endif
