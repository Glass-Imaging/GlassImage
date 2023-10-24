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

#ifndef gls_h
#define gls_h

#include <metal_stdlib>
using namespace metal;

#define TEXTURE(id) [[texture(id)]]
#define BUFFER(id) [[buffer(id)]]

#define USE_GLOBAL_ID           , uint2 __gls__global_id [[thread_position_in_grid]]
#define get_global_id(dim)      __gls__global_id[dim]

#define USE_GLOBAL_SIZE         , uint2 __gls__global_size [[threads_per_grid]]
#define get_global_size(dim)    __gls__global_size[dim]

#define USE_LOCAL_ID            , uint2 __gls__local_id [[thread_position_in_threadgroup]]
#define get_local_id(dim)       __gls__local_id[dim]

#define USE_GROUP_ID            , uint2 __gls__group_id [[threadgroup_position_in_grid]]
#define get_group_id(dim)       __gls__group_id[dim]

#define VAL(type)       constant type&

#define global device
#define private thread
#define local threadgroup

#define __ovld

typedef texture2d<float> image2df_t;
typedef texture2d<float, access::write> image2df_write_t;

typedef texture2d<half> image2dh_t;
typedef texture2d<half, access::write> image2dh_write_t;

typedef texture2d<uint> image2dui_t;
typedef texture2d<uint, access::write> image2dui_write_t;

template <typename T, access a>
static inline int get_image_width(texture2d<T, a> texture) {
    return texture.get_width();
}

template <typename T, access a>
static inline int get_image_height(texture2d<T, a> texture) {
    return texture.get_height();
}

template <typename T, access a>
static inline int2 get_image_dim(texture2d<T, a> image) {
    return int2(image.get_width(), image.get_height());
}

template <typename T>
static inline T sincos(T x, private T *cosval) {
    return sincos(x, *cosval);
}

#define convert_uint2(x)    uint2(x)
#define convert_uint3(x)    uint3(x)
#define convert_uint4(x)    uint4(x)

#define convert_int2(x)     int2(x)
#define convert_int3(x)     int3(x)
#define convert_int4(x)     int4(x)

#define convert_half2(x)    half2(x)
#define convert_half3(x)    half3(x)
#define convert_half4(x)    half4(x)

#define convert_float2(x)   float2(x)
#define convert_float3(x)   float3(x)
#define convert_float4(x)   float4(x)

#define as_uint2(x)    uint2(x)
#define as_uint3(x)    uint3(x)
#define as_uint4(x)    uint4(x)

#define as_int2(x)     int2(x)
#define as_int3(x)     int3(x)
#define as_int4(x)     int4(x)

#define as_half2(x)    half2(x)
#define as_half3(x)    half3(x)
#define as_half4(x)    half4(x)
#define as_half8(x)    _half8(x)

#define as_float2(x)   float2(x)
#define as_float3(x)   float3(x)
#define as_float4(x)   float4(x)

static inline float4 read_imagef(image2df_t image, sampler s, float2 coord) {
    return image.sample(s, coord);
}

static inline float4 read_imagef(image2df_t image, sampler s, int2 coord) {
    uint2 dim = uint2(get_image_width(image), get_image_height(image));
    return image.sample(s, float2(coord) / float2(dim));
}

static inline float4 read_imagef(image2df_t image, int2 coord) {
    return image.read(static_cast<uint2>(coord));
}

static inline void write_imagef(image2df_write_t image, int2 coord, float4 value) {
    image.write(value, static_cast<uint2>(coord));
}

static inline void write_imageh(image2dh_write_t image, int2 coord, half4 value) {
    image.write(value, static_cast<uint2>(coord));
}

static inline half4 read_imageh(image2dh_t image, int2 coord) {
    return image.read(static_cast<uint2>(coord));
}

static inline half4 read_imageh(image2dh_t image, sampler s, float2 coord) {
    return image.sample(s, coord);
}

static inline void write_imageui(image2dui_write_t image, int2 coord, uint4 value) {
    image.write(value, static_cast<uint2>(coord));
}

static inline uint4 read_imageui(image2dui_t image, int2 coord) {
    return image.read(static_cast<uint2>(coord));
}

static inline uint4 read_imageui(image2dui_t image, sampler s, float2 coord) {
    return image.sample(s, coord);
}

#endif /* gls_h */
