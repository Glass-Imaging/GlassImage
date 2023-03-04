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

#ifndef GLS_IMAGE_H
#define GLS_IMAGE_H

#include <sys/types.h>
#include <string.h>

#include <cassert>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "gls_geometry.hpp"
#include "gls_image_jpeg.h"
#include "gls_image_png.h"
#include "gls_image_tiff.h"

#if defined(__linux__) && !defined(__ANDROID__)
#define __TRUE_LINUX__
#else
#if !defined(__x86_64__)
#define USE_FP16_FLOATS true
#endif
#endif

namespace gls {

template <typename T, int N>
struct pixel : public std::array<T, N> {
    constexpr const static size_t channels = N;
    constexpr const static int bit_depth = 8 * sizeof(T);
    typedef T value_type;
};

template <typename T>
struct luma_type {
    typedef pixel<T, 1> pixel_type;
    union {
        pixel_type v;
        struct {
            T luma;
        };
        struct {
            T x;
        };
    };

    constexpr operator T() const { return v[0]; }
};

template <typename T>
struct luma_alpha_type {
    typedef pixel<T, 2> pixel_type;
    union {
        pixel_type v;
        struct {
            T luma, alpha;
        };
        struct {
            T x, y;
        };
    };
};

template <typename T>
struct rgb_type {
    typedef pixel<T, 3> pixel_type;
    union {
        pixel_type v;
        struct {
            T red, green, blue;
        };
        struct {
            T x, y, z;
        };
    };
};

template <typename T>
struct rgba_type {
    typedef pixel<T, 4> pixel_type;
    union {
        pixel_type v;
        struct {
            T red, green, blue, alpha;
        };
        struct {
            T x, y, z, w;
        };
    };
};

template <typename T>
struct basic_pixel : public T {
    constexpr static size_t channels = T::pixel_type::channels;
    constexpr static int bit_depth = T::pixel_type::bit_depth;
    typedef typename T::pixel_type::value_type value_type;

    constexpr basic_pixel() {}
    constexpr basic_pixel(value_type value) {
        static_assert(channels == 1);
        this->v[0] = value;
    }
    constexpr basic_pixel(value_type _v[channels]) { std::copy(&_v[0], &_v[channels], this->v.begin()); }
    constexpr basic_pixel(const std::array<value_type, channels>& _v) {
        std::copy(_v.begin(), _v.end(), this->v.begin());
    }

    template <typename T2>
    constexpr basic_pixel(const std::array<T2, channels>& _v) {
        for (int i = 0; i < channels; i++) {
            this->v[i] = _v[i];
        }
    }

    constexpr basic_pixel(std::initializer_list<value_type> list) {
        assert(list.size() == channels);
        std::copy(list.begin(), list.end(), this->v.begin());
    }

    constexpr value_type& operator[](int c) { return this->v[c]; }
    constexpr const value_type& operator[](int c) const { return this->v[c]; }
};

template <typename T>
constexpr basic_pixel<T> lerp(const basic_pixel<T>& p1, const basic_pixel<T>& p2, float alpha) {
    basic_pixel<T> result;
    for (int c = 0; c < T::pixel_type::channels; c++) {
        result[c] = (typename T::pixel_type::value_type)(p1[c] + alpha * (p2[c] - p1[c]));
    }
    return result;
}

typedef basic_pixel<luma_type<uint8_t>> luma_pixel;
typedef basic_pixel<luma_alpha_type<uint8_t>> luma_alpha_pixel;
typedef basic_pixel<rgb_type<uint8_t>> rgb_pixel;
typedef basic_pixel<rgba_type<uint8_t>> rgba_pixel;

typedef basic_pixel<luma_type<uint16_t>> luma_pixel_16;
typedef basic_pixel<luma_alpha_type<uint16_t>> luma_alpha_pixel_16;
typedef basic_pixel<rgb_type<uint16_t>> rgb_pixel_16;
typedef basic_pixel<rgba_type<uint16_t>> rgba_pixel_16;

typedef basic_pixel<luma_type<float>> luma_pixel_fp32;
typedef basic_pixel<luma_alpha_type<float>> luma_alpha_pixel_fp32;
typedef basic_pixel<rgb_type<float>> rgb_pixel_fp32;
typedef basic_pixel<rgba_type<float>> rgba_pixel_fp32;

typedef basic_pixel<luma_type<float>> pixel_fp32;
typedef basic_pixel<luma_alpha_type<float>> pixel_fp32_2;
typedef basic_pixel<rgb_type<float>> pixel_fp32_3;
typedef basic_pixel<rgba_type<float>> pixel_fp32_4;

#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
typedef __fp16 float16_t;
typedef basic_pixel<luma_type<float16_t>> luma_pixel_fp16;
typedef basic_pixel<luma_alpha_type<float16_t>> luma_alpha_pixel_fp16;
typedef basic_pixel<rgb_type<float16_t>> rgb_pixel_fp16;
typedef basic_pixel<rgba_type<float16_t>> rgba_pixel_fp16;
#endif

#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
typedef float16_t float_type;
#else
typedef float float_type;
#endif
typedef basic_pixel<luma_type<float_type>> luma_pixel_float;
typedef basic_pixel<luma_alpha_type<float_type>> luma_alpha_pixel_float;
typedef basic_pixel<rgb_type<float_type>> rgb_pixel_float;
typedef basic_pixel<rgba_type<float_type>> rgba_pixel_float;

typedef basic_pixel<luma_type<float_type>> pixel_float;
typedef basic_pixel<luma_alpha_type<float_type>> pixel_float2;
typedef basic_pixel<rgb_type<float_type>> pixel_float3;
typedef basic_pixel<rgba_type<float_type>> pixel_float4;

class tiff_metadata;

template <typename T>
class basic_image {
   public:
    const int width;
    const int height;

    constexpr gls::size size() const { return {width, height}; }

    typedef T pixel_type;
    typedef std::unique_ptr<basic_image<T>> unique_ptr;

    static const constexpr int bit_depth = pixel_type::bit_depth;
    static const constexpr int channels = pixel_type::channels;
    static const constexpr int pixel_size = sizeof(pixel_type);

    constexpr basic_image(int _width, int _height) : width(_width), height(_height) {}
    constexpr basic_image(gls::size _dimensions) : width(_dimensions.width), height(_dimensions.height) {}
};

// std::vector is convenient but it is expensive as it initializes memory
#define USE_STD_VECTOR_ALLOCATION false

template <typename T>
class image : public basic_image<T> {
   public:
    const int stride;
    typedef std::unique_ptr<image<T>> unique_ptr;

   protected:
#if USE_STD_VECTOR_ALLOCATION
    const std::unique_ptr<std::vector<T>> _data_store = nullptr;
#else
    T* _data_store = nullptr;
#endif
    const std::span<T> _data;

   public:
    // Data is owned by the image and retained by _data_store
    constexpr image(int _width, int _height, int _stride)
        : basic_image<T>(_width, _height),
          stride(_stride),
#if USE_STD_VECTOR_ALLOCATION
          _data_store(std::make_unique<std::vector<T>>(_stride * _height)),
          _data(_data_store->data(), _data_store->size()) {
    }
#else
          _data_store(new T[_stride * _height]),
          _data(_data_store, _stride * _height) {
    }

    virtual ~image() { delete _data_store; }
#endif

    constexpr image(int _width, int _height) : image(_width, _height, _width) {}

    constexpr image(size _dimensions) : image(_dimensions.width, _dimensions.height) {}

    // Data is owned by caller, the image is only a wrapper around it
    constexpr image(int _width, int _height, int _stride, std::span<T> data)
        : basic_image<T>(_width, _height), stride(_stride), _data(data) {
        assert(_stride * _height <= data.size());
    }

    constexpr image(int _width, int _height, std::span<T> data) : image<T>(_width, _height, _width, data) {}

    constexpr image(image* _base, int _x, int _y, int _width, int _height)
        : image<T>(_width, _height, _base->stride,
                   std::span(_base->_data.data() + _y * _base->stride + _x, _base->stride * _height)) {
        assert(_x + _width <= _base->width && _y + _height <= _base->height);
    }

    constexpr image(image* _base, const rectangle& _crop) : image(_base, _crop.x, _crop.y, _crop.width, _crop.height) {}

    constexpr image(const image& _base, int _x, int _y, int _width, int _height)
        : image<T>(_width, _height, _base.stride,
                   std::span(_base._data.data() + _y * _base.stride + _x, _base.stride * _height)) {
        assert(_x + _width <= _base.width && _y + _height <= _base.height);
    }

    constexpr image(const image& _base, const rectangle& _crop)
        : image(_base, _crop.x, _crop.y, _crop.width, _crop.height) {}

    // row access
    constexpr T* operator[](int row) { return &_data[stride * row]; }

    const constexpr T* operator[](int row) const { return &_data[stride * row]; }

    const constexpr std::span<T> pixels() const { return _data; }

    constexpr const T& getPixel(int x, int y) const {
        if (y < 0) {
            y = std::min(-y, basic_image<T>::height-1);
        } else if (y > basic_image<T>::height-1) {
            y = 2 * (basic_image<T>::height-1) - y;
        }
        if (x < 0) {
            x = std::min(-x, basic_image<T>::width-1);
        } else if (x > basic_image<T>::width-1) {
            x = 2 * (basic_image<T>::width-1) - x;
        }
        return (*this)[y][x];
    }

    constexpr void apply(std::function<void(const T& pixel)> process) const {
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process((*this)[y][x]);
            }
        }
    }

    constexpr void apply(std::function<void(const T& pixel, int x, int y)> process) const {
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process((*this)[y][x], x, y);
            }
        }
    }

    constexpr void apply(std::function<void(T* pixel, int x, int y)> process) {
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process(&(*this)[y][x], x, y);
            }
        }
    }

    const constexpr size_t size_in_bytes() { return _data.size() * basic_image<T>::pixel_size; }

    // image factory from PNG file
    constexpr static unique_ptr read_png_file(const std::string& filename) {
        unique_ptr image = nullptr;

        auto image_allocator = [&image](int width, int height, std::vector<uint8_t*>* row_pointers) -> bool {
            if ((image = std::make_unique<gls::image<T>>(width, height)) == nullptr) {
                return false;
            }
            for (int i = 0; i < height; ++i) {
                (*row_pointers)[i] = (uint8_t*)(*image)[i];
            }
            return true;
        };

        gls::read_png_file(filename, T::channels, T::bit_depth, image_allocator);

        return image;
    }

    // Write image to PNG file
    // compression_level range: [0-9], 0 -> no compression (default), 1 -> *fast* compression, otherwise useful range:
    // [3-6]
    constexpr void write_png_file(const std::string& filename, int compression_level = 0) const {
        auto row_pointer = [this](int row) -> uint8_t* { return (uint8_t*)(*this)[row]; };
        gls::write_png_file(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth, false,
                            compression_level, row_pointer);
    }

    constexpr void write_png_file(const std::string& filename, bool skip_alpha, int compression_level = 0) const {
        auto row_pointer = [this](int row) -> uint8_t* { return (uint8_t*)(*this)[row]; };
        gls::write_png_file(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth,
                            skip_alpha, compression_level, row_pointer);
    }

    // Image factory from JPEG file
    constexpr static unique_ptr read_jpeg_file(const std::string& filename) {
        static_assert(basic_image<T>::channels == 1 || basic_image<T>::channels == 3,
                      "The JPEG codec only supports 1-channel or 3-channel images.");

        unique_ptr image = nullptr;

        auto image_allocator = [&image](int width, int height) -> std::span<uint8_t> {
            if ((image = std::make_unique<gls::image<T>>(width, height)) == nullptr) {
                return std::span<uint8_t>();
            }
            return std::span<uint8_t>((uint8_t*)(*image)[0], sizeof(T) * width * height);
        };

        gls::read_jpeg_file(filename, T::channels, T::bit_depth, image_allocator);

        return image;
    }

    // Write image to JPEG file
    constexpr void write_jpeg_file(const std::string& filename, int quality) const {
        static_assert(basic_image<T>::channels == 1 || basic_image<T>::channels == 3,
                      "The JPEG codec only supports 1-channel or 3-channel images.");

        auto image_data = [this]() -> std::span<uint8_t> {
            return std::span<uint8_t>((uint8_t*)this->_data.data(), sizeof(T) * this->_data.size());
        };
        gls::write_jpeg_file(filename, basic_image<T>::width, basic_image<T>::height, stride, T::channels, T::bit_depth,
                             image_data, quality);
    }

    // Helper function for read_tiff_file and read_dng_file
    constexpr static bool process_tiff_strip(image* destination, int tiff_bitspersample, int tiff_samplesperpixel,
                                             int destination_row, int strip_width, int strip_height, int crop_x,
                                             int crop_y, uint8_t* tiff_buffer) {
        typedef typename T::value_type value_type;

        std::function<value_type()> nextTiffPixelSame = [&tiff_buffer]() -> value_type {
            value_type pixelValue = *((value_type*)tiff_buffer);
            tiff_buffer += sizeof(value_type);
            return pixelValue;
        };
        std::function<value_type()> nextTiffPixel8to16 = [&tiff_buffer]() -> value_type {
            return (value_type) * (tiff_buffer++) << 8;
            ;
        };
        std::function<value_type()> nextTiffPixel16to8 = [&tiff_buffer]() -> value_type {
            value_type pixelValue = (value_type)(*((uint16_t*)tiff_buffer) >> 8);
            tiff_buffer += sizeof(uint16_t);
            return pixelValue;
        };

        auto nextTiffPixel = tiff_bitspersample == T::bit_depth ? nextTiffPixelSame
                             : (tiff_bitspersample == 8)        ? nextTiffPixel8to16
                                                                : nextTiffPixel16to8;

        for (int y = 0; y < strip_height && y + destination_row - crop_y < destination->height; y++) {
            for (int x = 0; x < strip_width; x++) {
                for (int c = 0; c < std::min((int)tiff_samplesperpixel, (int)T::channels); c++) {
                    if (x >= crop_x && y + destination_row >= crop_y && x - crop_x < destination->width) {
                        (*destination)[y + destination_row - crop_y][x - crop_x][c] = nextTiffPixel();
                    } else {
                        nextTiffPixel();
                    }
                }
            }
        }
        return true;
    };

    // Image factory from TIFF file
    constexpr static unique_ptr read_tiff_file(const std::string& filename, tiff_metadata* metadata = nullptr) {
        unique_ptr image = nullptr;
        gls::read_tiff_file(
            filename, T::channels, T::bit_depth, metadata,
            [&image](int width, int height) -> bool {
                return (image = std::make_unique<gls::image<T>>(width, height)) != nullptr;
            },
            [&image](int tiff_bitspersample, int tiff_samplesperpixel, int row, int strip_width, int strip_height,
                     int crop_x, int crop_y, uint8_t* tiff_buffer) -> bool {
                return process_tiff_strip(image.get(), tiff_bitspersample, tiff_samplesperpixel, row,
                                          /*strip_width=*/image->width, strip_height,
                                          /*crop_x=*/0, /*crop_y=*/0, tiff_buffer);
            });
        return image;
    }

    // Write image to TIFF file
    constexpr void write_tiff_file(const std::string& filename, tiff_compression compression = tiff_compression::NONE,
                                   tiff_metadata* metadata = nullptr) const {
        typedef typename T::value_type value_type;
        auto row_pointer = [this](int row) -> value_type* { return (value_type*)(*this)[row]; };
        gls::write_tiff_file<value_type>(filename, basic_image<T>::width, basic_image<T>::height, T::channels,
                                         T::bit_depth, compression, metadata, row_pointer);
    }

    // Image factory from DNG file
    constexpr static unique_ptr read_dng_file(const std::string& filename, tiff_metadata* dng_metadata = nullptr,
                                              tiff_metadata* exif_metadata = nullptr) {
        unique_ptr image = nullptr;
        gls::read_dng_file(
            filename, T::channels, T::bit_depth, dng_metadata, exif_metadata,
            [&image](int width, int height) -> bool {
                return (image = std::make_unique<gls::image<T>>(width, height)) != nullptr;
            },
            [&image](int tiff_bitspersample, int tiff_samplesperpixel, int row, int strip_width, int strip_height,
                     int crop_x, int crop_y, uint8_t* tiff_buffer) -> bool {
                return process_tiff_strip(image.get(), tiff_bitspersample, tiff_samplesperpixel, row,
                                          /*strip_width=*/strip_width, strip_height,
                                          /*crop_x=*/crop_x, /*crop_y=*/crop_y, tiff_buffer);
            });
        return image;
    }

    // Write image to DNG file
    constexpr void write_dng_file(const std::string& filename, tiff_compression compression = tiff_compression::NONE,
                                  const tiff_metadata* dng_metadata = nullptr,
                                  const tiff_metadata* exif_metadata = nullptr) const {
        typedef typename T::value_type value_type;
        auto row_pointer = [this](int row) -> value_type* { return (value_type*)(*this)[row]; };
        gls::write_dng_file(filename, basic_image<T>::width, basic_image<T>::height, T::channels, T::bit_depth,
                            compression, dng_metadata, exif_metadata, row_pointer);
    }
};

template <typename T>
constexpr static inline void copyPixels(gls::image<T>* to, const gls::image<T>& from) {
    assert(to->width == from.width && to->height == from.height);

    if (to->stride == from.stride) {
        memcpy((void*)(*to)[0], (void*)from[0], to->stride * to->height * sizeof(T));
    } else {
        for (int j = 0; j < to->height; j++) {
            memcpy((void*)(*to)[j], (void*)from[j], to->width * sizeof(T));
        }
    }
}

}  // namespace gls

#endif /* GLS_IMAGE_H */
