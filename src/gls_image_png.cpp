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

#include <assert.h>
#include <png.h>
#include <zlib.h>

#include "gls_image_png.h"

namespace gls {

void read_png_file(const std::string& filename, int pixel_channels, int pixel_bit_depth,
                   std::function<bool(int width, int height, std::vector<uint8_t*>* row_pointers)> image_allocator) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Could not open " + filename);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        throw std::runtime_error("Could not create png read struct " + filename);
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        throw std::runtime_error("Could not create png info struct " + filename);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(fp);
        throw std::runtime_error("Error reading PNG file: " + filename);
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 png_width, png_height;
    int png_color_type, png_bit_depth;
    png_get_IHDR(png_ptr, info_ptr, &png_width, &png_height, &png_bit_depth, &png_color_type, nullptr, nullptr,
                 nullptr);

    // Match the image's data layout
    if ((pixel_channels == 4 && png_color_type == PNG_COLOR_TYPE_RGB) ||
        (pixel_channels == 2 && png_color_type == PNG_COLOR_TYPE_GRAY)) {
        png_set_add_alpha(png_ptr, 0, PNG_FILLER_AFTER);
    }
    if ((pixel_channels > 1 && png_color_type == PNG_COLOR_TYPE_GRAY) ||
        (pixel_channels > 2 && png_color_type == PNG_COLOR_TYPE_GRAY_ALPHA)) {
        png_set_gray_to_rgb(png_ptr);
    }
    if ((pixel_channels != 4 && pixel_channels != 2) && (png_color_type & PNG_COLOR_MASK_ALPHA)) {
        png_set_strip_alpha(png_ptr);
    }
    if (pixel_bit_depth == 8 && png_bit_depth == 16) {
        png_set_strip_16(png_ptr);
    }
    if (pixel_bit_depth == 16 && png_bit_depth == 8) {
        png_set_expand_16(png_ptr);
    }

#if __LITTLE_ENDIAN__
    png_set_swap(png_ptr);
#endif
    png_read_update_info(png_ptr, info_ptr);

#if (defined(__ANDROID__) && !defined(NDEBUG))
    size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    assert(rowbytes == png_width * pixel_channels * pixel_bit_depth / 8);
#endif
    std::vector<uint8_t*> row_pointers(png_height);
    if (image_allocator(png_width, png_height, &row_pointers)) {
        png_read_image(png_ptr, row_pointers.data());
    }
    png_read_end(png_ptr, nullptr);

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(fp);
}

void write_png_file(const std::string& filename, int width, int height, int pixel_channels, int pixel_bit_depth,
                    bool skip_alpha, int compression_level, std::function<uint8_t*(int row)> row_pointer) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Could not open " + filename);
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        throw std::runtime_error("Could not create png write struct " + filename);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        throw std::runtime_error("Could not create png info struct " + filename);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        throw std::runtime_error("Error writing PNG file: " + filename);
    }

    png_init_io(png_ptr, fp);

    int png_color_type = PNG_COLOR_TYPE_RGB;
    if (pixel_channels == 1)
        png_color_type = PNG_COLOR_TYPE_GRAY;
    else if (pixel_channels == 2)
        png_color_type = skip_alpha ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_GRAY_ALPHA;
    else if (pixel_channels == 3)
        png_color_type = PNG_COLOR_TYPE_RGB;
    else if (pixel_channels == 4)
        png_color_type = skip_alpha ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGB_ALPHA;

    png_set_IHDR(png_ptr, info_ptr, width, height, pixel_bit_depth, png_color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Fast compression strategy with fast filtering.
    // Save time: 10x faster on Android with ~10% worse compression
    if (compression_level <= 1) {
        png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FAST_FILTERS);
        png_set_compression_strategy(png_ptr, Z_RLE);
    }
    // Use larger zlib buffer for speed
    png_set_compression_mem_level(png_ptr, 9);

    png_set_compression_level(png_ptr, compression_level);

    png_write_info(png_ptr, info_ptr);

    if (skip_alpha && (pixel_channels == 2 || pixel_channels == 4)) {
        png_set_filler(png_ptr, 0, PNG_FILLER_AFTER);
    }

#if __LITTLE_ENDIAN__
    png_set_swap(png_ptr);
#endif

    std::vector<uint8_t*> row_pointers(height);
    for (int i = 0; i < height; ++i) {
        row_pointers[i] = row_pointer(i);
    }
    png_write_image(png_ptr, row_pointers.data());

    png_write_end(png_ptr, nullptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

}  // namespace gls
