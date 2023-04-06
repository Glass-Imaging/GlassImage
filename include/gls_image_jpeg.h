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

#ifndef GLS_IMAGE_JPEG_HPP
#define GLS_IMAGE_JPEG_HPP

#include <functional>
#include <span>
#include <string>

#if defined(__linux__) && !defined(__ANDROID__)
#include <memory>
#include <stdexcept>
#endif

namespace gls {

void read_jpeg_file(const std::string& filename, int pixel_channels, int pixel_bit_depth,
                    std::function<std::span<uint8_t>(int width, int height)> image_allocator);

void write_jpeg_file(const std::string& fileName, int width, int height, int stride, int pixel_channels,
                     int pixel_bit_depth, const std::function<std::span<uint8_t>()>& image_data, int quality);

}  // namespace gls
#endif /* GLS_IMAGE_JPEG_HPP */
