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

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

float3 boxBlur(image2d_t blurMap, int2 imageCoordinates);

float3 boxBlur(image2d_t blurMap, int2 imageCoordinates) {
    const int filterSize = 15;
    float3 blur = 0;
    for (int y = -filterSize / 2; y <= filterSize / 2; y++) {
        for (int x = -filterSize / 2; x <= filterSize / 2; x++) {
            int2 sampleCoordinate = imageCoordinates + (int2)(x, y);
            float3 blurSample = read_imagef(blurMap, sampler, sampleCoordinate).xyz;
            blur += blurSample;
        }
    }
    return blur / (filterSize * filterSize);
}

kernel void blur(read_only image2d_t input, write_only image2d_t output) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    float3 result = boxBlur(input, imageCoordinates);
    write_imagef(output, imageCoordinates, (float4) (result, 1));
}
