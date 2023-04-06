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

#ifndef GLS_IMAGE_PNG_H
#define GLS_IMAGE_PNG_H

#include <functional>
#include <string>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <stdexcept>
#endif

namespace gls {

void read_png_file(const std::string& filename, int pixel_channels, int pixel_bit_depth,
                   std::function<bool(int width, int height, std::vector<uint8_t*>* row_pointers)> image_allocator);

void write_png_file(const std::string& filename, int width, int height, int pixel_channels, int pixel_bit_depth,
                    bool skip_alpha, int compression_level, const std::vector<uint8_t>* icc_profile_data,
                    std::function<uint8_t*(int row)> row_pointer);

}  // namespace gls

#endif /* GLS_IMAGE_PNG_H */
