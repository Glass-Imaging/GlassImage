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

#include "gls_image_jpeg.h"

#include <jpeglib.h>

#include <cassert>

namespace gls {

void read_jpeg_file(const std::string& filename, int pixel_channels, int pixel_bit_depth,
                    std::function<std::span<uint8_t>(int width, int height)> image_allocator) {
    if ((pixel_channels != 3 && pixel_channels != 1) || pixel_bit_depth != 8) {
        throw std::runtime_error("Can only create JPEG files for 8-bit RGB or Grayscale images");
    }

    // Creating a custom deleter for the decompressInfo pointer
    // to ensure ::jpeg_destroy_compress() gets called even if
    // we throw out of this function.
    auto dt = [](::jpeg_decompress_struct* ds) { ::jpeg_destroy_decompress(ds); };
    std::unique_ptr<::jpeg_decompress_struct, decltype(dt)> decompressInfo(new ::jpeg_decompress_struct, dt);

    // Note this is a shared pointer as we can share this
    // between objects which have copy constructed from each other
    auto errorMgr = std::make_shared<::jpeg_error_mgr>();

    // Using fopen here ( and in save() ) because libjpeg expects
    // a FILE pointer.
    // We store the FILE* in a unique_ptr so we can also use the custom
    // deleter here to ensure fclose() gets called even if we throw.
    auto fdt = [](FILE* fp) { fclose(fp); };
    std::unique_ptr<FILE, decltype(fdt)> infile(fopen(filename.c_str(), "rb"), fdt);
    if (infile.get() == nullptr) {
        throw std::runtime_error("Could not open " + filename);
    }

    decompressInfo->err = ::jpeg_std_error(errorMgr.get());
    // Note this usage of a lambda to provide our own error handler
    // to libjpeg. If we do not supply a handler, and libjpeg hits
    // a problem, it just prints the error message and calls exit().
    errorMgr->error_exit = [](::j_common_ptr cinfo) {
        char jpegLastErrorMsg[JMSG_LENGTH_MAX];
        // Call the function pointer to get the error message
        (*(cinfo->err->format_message))(cinfo, jpegLastErrorMsg);
        throw std::runtime_error(jpegLastErrorMsg);
    };
    ::jpeg_create_decompress(decompressInfo.get());

    // Read the file:
    ::jpeg_stdio_src(decompressInfo.get(), infile.get());

    int rc = ::jpeg_read_header(decompressInfo.get(), TRUE);
    if (rc != 1) {
        throw std::runtime_error("File does not seem to be a normal JPEG");
    }
    ::jpeg_start_decompress(decompressInfo.get());

    int width = decompressInfo->output_width;
    int height = decompressInfo->output_height;
    int pixelSize = decompressInfo->output_components;
    // int colourSpace = decompressInfo->out_color_space;

    if (pixelSize != pixel_channels) {
        throw std::runtime_error("Pixel size " + std::to_string(pixelSize) + " doesn't match the image's channels " +
                                 std::to_string(pixel_channels));
    }

    size_t row_stride = width * pixelSize;

    std::span<uint8_t> imageData = image_allocator(width, height);

    assert(imageData.size() == row_stride * height && imageData.data() != nullptr);

    if (imageData.size() != row_stride * height || imageData.data() == nullptr) {
        throw std::runtime_error("Image allocation failed");
    }

    uint8_t* ptr = imageData.data();

    while (decompressInfo->output_scanline < height) {
        ::jpeg_read_scanlines(decompressInfo.get(), &ptr, 1);
        ptr += row_stride;
    }
    ::jpeg_finish_decompress(decompressInfo.get());
}

void write_jpeg_file(const std::string& fileName, int width, int height, int stride, int pixel_channels,
                     int pixel_bit_depth, const std::function<std::span<uint8_t>()>& image_data, int quality) {
    if ((pixel_channels != 3 && pixel_channels != 1) || pixel_bit_depth != 8) {
        throw std::runtime_error("Can only create JPEG files for 8-bit RGB or Grayscale images");
    }

    if (quality < 0) {
        quality = 0;
    }
    if (quality > 100) {
        quality = 100;
    }
    FILE* outfile = fopen(fileName.c_str(), "wb");
    if (outfile == nullptr) {
        throw std::runtime_error("Could not open " + fileName + " for writing");
    }

    auto errorMgr = std::make_shared<::jpeg_error_mgr>();

    // Creating a custom deleter for the compressInfo pointer
    // to ensure ::jpeg_destroy_compress() gets called even if
    // we throw out of this function.
    auto dt = [](::jpeg_compress_struct* cs) { ::jpeg_destroy_compress(cs); };
    std::unique_ptr<::jpeg_compress_struct, decltype(dt)> compressInfo(new ::jpeg_compress_struct, dt);
    ::jpeg_create_compress(compressInfo.get());
    ::jpeg_stdio_dest(compressInfo.get(), outfile);
    compressInfo->image_width = (JDIMENSION)width;
    compressInfo->image_height = (JDIMENSION)height;
    compressInfo->input_components = (JDIMENSION)pixel_channels;
    compressInfo->in_color_space = static_cast<::J_COLOR_SPACE>(pixel_channels == 3 ? ::JCS_RGB : ::JCS_GRAYSCALE);
    compressInfo->err = ::jpeg_std_error(errorMgr.get());
    ::jpeg_set_defaults(compressInfo.get());
    ::jpeg_set_quality(compressInfo.get(), quality, TRUE);
    ::jpeg_start_compress(compressInfo.get(), TRUE);

    uint8_t* ptr = image_data().data();
    size_t row_stride = stride * pixel_channels;

    for (int line = 0; line < height; line++) {
        ::jpeg_write_scanlines(compressInfo.get(), &ptr, 1);
        ptr += row_stride;
    }

    ::jpeg_finish_compress(compressInfo.get());
    fclose(outfile);
}

}  // namespace gls
