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

extern float3 boxBlur(image2df_t blurMap, int2 imageCoordinates, int filterSize);

kernel void blur(image2df_t input               TEXTURE(0),
                 image2df_write_t output        TEXTURE(1),
                 constant int *filterSize       BUFFER(2)
                 USE_GLOBAL_ID
                 ) {
    const int2 imageCoordinates = int2(get_global_id(0), get_global_id(1));
    float3 result = boxBlur(input, imageCoordinates, *filterSize);
    write_imagef(output, imageCoordinates, float4(result, 1));
}
