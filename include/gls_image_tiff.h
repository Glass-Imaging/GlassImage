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

#ifndef gls_image_tiff_hpp
#define gls_image_tiff_hpp

#include <functional>
#include <span>
#include <string>

namespace gls {

// clang-format off

typedef enum tiff_compression {
    NONE            = 1,        /* dump mode */
    LZW             = 5,        /* Lempel-Ziv & Welch */
    JPEG            = 7,        /* %JPEG DCT compression */
    PACKBITS        = 32773,    /* Macintosh RLE */
    DEFLATE         = 32946,    /* Deflate compression */
    ADOBE_DEFLATE   = 8,        /* Deflate compression, as recognized by Adobe */
} tiff_compression;

// clang-format on

class tiff_metadata;

typedef std::function<bool(int tiff_bitspersample, int tiff_samplesperpixel, int row, int strip_width, int strip_height,
                           int crop_x, int crop_y, uint8_t* tiff_buffer)>
    tiff_strip_procesor;

void read_tiff_file(const std::string& filename, int pixel_channels, int pixel_bit_depth, tiff_metadata* metadata,
                    std::function<bool(int width, int height)> image_allocator, tiff_strip_procesor process_tiff_strip);

template <typename T>
void write_tiff_file(const std::string& filename, int width, int height, int pixel_channels, int pixel_bit_depth,
                     tiff_compression compression, tiff_metadata* metadata, const std::vector<uint8_t>* icc_profile_data,
                     std::function<T*(int row)> row_pointer);

void read_dng_file(const std::string& filename, int pixel_channels, int pixel_bit_depth, tiff_metadata* dng_metadata,
                   tiff_metadata* exif_metadata, std::function<bool(int width, int height)> image_allocator,
                   tiff_strip_procesor process_tiff_strip);

void write_dng_file(const std::string& filename, int width, int height, int pixel_channels, int pixel_bit_depth,
                    tiff_compression compression, const tiff_metadata* dng_metadata, const tiff_metadata* exif_metadata,
                    std::function<uint16_t*(int row)> row_pointer);

}  // namespace gls

#endif /* gls_image_tiff_hpp */
