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

#include "gls.h"

#ifdef __METAL__
constexpr sampler linear_sampler(filter::linear, address::clamp_to_edge);
#else
const sampler_t linear_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
#endif

float3 boxBlur(image2df_t blurMap, int2 imageCoordinates, int filterSize) {
    float2 norm = 1 / float2(get_image_width(blurMap), get_image_height(blurMap));

    // const int filterSize = 15;
    float3 blur = 0;
    for (int y = -filterSize / 2; y <= filterSize / 2; y++) {
        for (int x = -filterSize / 2; x <= filterSize / 2; x++) {
            float2 samplePos = (convert_float2(imageCoordinates) + float2(x, y) + 0.5) * norm;
            float3 blurSample = read_imagef(blurMap, linear_sampler, samplePos).xyz;
            blur += blurSample;
        }
    }
    return blur / (filterSize * filterSize);
}
