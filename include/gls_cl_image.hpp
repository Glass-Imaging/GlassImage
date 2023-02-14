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

#ifndef CL_IMAGE_H
#define CL_IMAGE_H

#include <cassert>

#include "gls_cl.hpp"
#include "gls_image.hpp"

namespace gls {

template <typename T>
class cl_image : public basic_image<T> {
   public:
    typedef std::unique_ptr<cl_image<T>> unique_ptr;

    cl_image(int _width, int _height) : basic_image<T>(_width, _height) {}

    static inline cl::ImageFormat ImageFormat() {
        static_assert(T::channels == 1 || T::channels == 2 || T::channels == 4);
        static_assert(std::is_same<typename T::value_type, float>::value ||
#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
                      std::is_same<typename T::value_type, gls::float16_t>::value ||
#endif
                      std::is_same<typename T::value_type, uint8_t>::value ||
                      std::is_same<typename T::value_type, uint16_t>::value ||
                      std::is_same<typename T::value_type, uint32_t>::value ||
                      std::is_same<typename T::value_type, int32_t>::value);

        cl_channel_order order = T::channels == 1 ? CL_R : T::channels == 2 ? CL_RG : CL_RGBA;
        cl_channel_type type = std::is_same<typename T::value_type, float>::value ? CL_FLOAT :
#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
                               std::is_same<typename T::value_type, gls::float16_t>::value ? CL_HALF_FLOAT :
#endif
                               std::is_same<typename T::value_type, uint8_t>::value ? CL_UNORM_INT8 :
                               std::is_same<typename T::value_type, uint16_t>::value ? CL_UNORM_INT16 :
                               std::is_same<typename T::value_type, uint32_t>::value ? CL_UNSIGNED_INT32 :
                               /* std::is_same<typename T::value_type, int32_t>::value ? */ CL_SIGNED_INT32;

        return cl::ImageFormat(order, type);
    }
};

// Other Supported OpenCL mappings

template <>
inline cl::ImageFormat cl_image<float>::ImageFormat() {
    return cl::ImageFormat(CL_R, CL_FLOAT);
}

template <>
inline cl::ImageFormat cl_image<std::array<float, 2>>::ImageFormat() {
    return cl::ImageFormat(CL_RG, CL_FLOAT);
}

template <>
inline cl::ImageFormat cl_image<std::array<float, 4>>::ImageFormat() {
    return cl::ImageFormat(CL_RGBA, CL_FLOAT);
}

#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
template <>
inline cl::ImageFormat cl_image<gls::float16_t>::ImageFormat() {
    return cl::ImageFormat(CL_R, CL_HALF_FLOAT);
}
#endif

template <>
inline cl::ImageFormat cl_image<uint8_t>::ImageFormat() {
    return cl::ImageFormat(CL_R, CL_UNORM_INT8);
}

template <>
inline cl::ImageFormat cl_image<uint16_t>::ImageFormat() {
    return cl::ImageFormat(CL_R, CL_UNORM_INT16);
}

template <>
inline cl::ImageFormat cl_image<uint32_t>::ImageFormat() {
    return cl::ImageFormat(CL_R, CL_UNSIGNED_INT32);
}

template <>
inline cl::ImageFormat cl_image<int32_t>::ImageFormat() {
    return cl::ImageFormat(CL_R, CL_SIGNED_INT32);
}

template <typename T>
class cl_image_2d : public cl_image<T> {
   protected:
    struct payload {
        const cl::Image2D image;
    };

    const std::unique_ptr<payload> _payload;

    cl_image_2d(cl::Context context, int _width, int _height, std::unique_ptr<payload> payload)
        : cl_image<T>(_width, _height), _payload(std::move(payload)) {}

   public:
    typedef std::unique_ptr<cl_image_2d<T>> unique_ptr;

    cl_image_2d(cl::Context context, int _width, int _height)
        : cl_image<T>(_width, _height), _payload(buildPayload(context, _width, _height)) {}

    cl_image_2d(cl::Context context, const gls::image<T>& other)
        : cl_image<T>(other.width, other.height),
          _payload(buildPayload(context, other.width, other.height))
    {
        copyPixelsFrom(other);
    }

    virtual ~cl_image_2d() {}

    static inline std::unique_ptr<payload> buildPayload(cl::Context context, int width, int height) {
        cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
        return std::make_unique<payload>( payload{cl::Image2D(context, mem_flags, cl_image<T>::ImageFormat(), width, height) });
    }

    static inline unique_ptr fromImage(cl::Context context, const gls::image<T>& other) {
        return std::make_unique<cl_image_2d<T>>(context, other);
    }

    inline typename gls::image<T>::unique_ptr toImage() const {
        auto image = std::make_unique<gls::image<T>>(gls::image<T>::width, gls::image<T>::height);
        copyPixelsTo(image.get());
        return image;
    }

    void copyPixelsFrom(const image<T>& other) const {
        assert(other.width == image<T>::width && other.height == image<T>::height);
        auto cpuImage = mapImage(CL_MAP_WRITE);
        copyPixels(&cpuImage, other);
        unmapImage(cpuImage);
    }

    void copyPixelsTo(image<T>* other) const {
        assert(other->width == image<T>::width && other->height == image<T>::height);
        auto cpuImage = mapImage(CL_MAP_READ);
        copyPixels(other, cpuImage);
        unmapImage(cpuImage);
    }

    virtual image<T> mapImage(cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE) const {
        size_t row_pitch;
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        T* image_data =
            (T*)queue.enqueueMapImage(_payload->image, CL_TRUE, map_flags, {0, 0, 0},
                                      {(size_t) image<T>::width, (size_t) image<T>::height, 1}, &row_pitch, /*slice_pitch=*/ nullptr);
        assert(image_data != nullptr);

        size_t stride = row_pitch / image<T>::pixel_size;
        size_t data_size = stride * image<T>::height;
        return gls::image(image<T>::width, image<T>::height, (int) stride, std::span<T>(image_data, data_size));
    }

    virtual void unmapImage(const image<T>& mappedImage) const {
        cl::enqueueUnmapMemObject(_payload->image, (void*)mappedImage[0]);
    }

    void apply(std::function<void(T* pixel, int x, int y)> process) {
        auto cpu_image = mapImage();
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process(&cpu_image[y][x], x, y);
            }
        }
        unmapImage(cpu_image);
    }

    cl::Image2D getImage2D() const { return _payload->image; }
};

template <typename T>
class cl_image_buffer_2d : public cl_image_2d<T> {
   private:
    struct payload : public cl_image_2d<T>::payload {
        const cl::Buffer buffer;
    };

    cl_image_buffer_2d(cl::Context context, int _width, int _height, int _stride)
        : cl_image_2d<T>(context, _width, _height, buildPayload(context, _width, _height, _stride)), stride(_stride) {}

   public:
    // typedef std::unique_ptr<cl_image_buffer_2d<T>> unique_ptr;

    const int stride;

    cl_image_buffer_2d(cl::Context context, int _width, int _height)
        : cl_image_buffer_2d<T>(context, _width, _height, compute_stride(_width)) {}

    cl_image_buffer_2d(cl::Context context, const gls::image<T>& other)
        : cl_image_buffer_2d(context, other.width, other.height) {
        this->copyPixelsFrom(other);
    }

    static int compute_stride(int width) {
        // pitch_alignment in *pixels*
        const int pitch_alignment = cl::Device::getDefault().getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();
        return pitch_alignment * ((width + pitch_alignment - 1) / pitch_alignment);
    }

    static inline std::unique_ptr<payload> buildPayload(cl::Context context, int width, int height, int stride) {
        auto buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, stride * height * sizeof(T));
        auto image = cl::Image2D(context, cl_image<T>::ImageFormat(), buffer, width, height, stride * sizeof(T));
        return std::make_unique<payload>(payload{{image}, buffer});
    }

    image<T> mapImage(cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE) const override {
        size_t pixel_count = stride * image<T>::height;
        T* image_data =
            (T *) cl::enqueueMapBuffer(getBuffer(), true, map_flags, 0, image<T>::pixel_size * pixel_count);

        return gls::image(image<T>::width, image<T>::height, stride, std::span<T>(image_data, pixel_count));
    }

    void unmapImage(const image<T>& mappedImage) const override { cl::enqueueUnmapMemObject(getBuffer(), (void*)mappedImage[0]); }

    cl::Buffer getBuffer() const { return static_cast<const payload*>(this->_payload.get())->buffer; }
};

template <typename T>
class cl_image_2d_array : public cl_image<T> {
    const cl::Image2DArray _image;

   public:
    const int depth;

    typedef std::unique_ptr<cl_image_2d_array<T>> unique_ptr;

    cl_image_2d_array(cl::Context context, int _width, int _height, int _depth)
        : cl_image<T>(_width, _height), depth(_depth), _image(buildImage(context, _width, _height, _depth)) {}

    cl_image_2d_array(cl::Context context, const gls::image<T>& other, int _depth) : cl_image_2d_array(other.width, other.height / _depth, _depth) {
        copyPixelsFrom(other);
    }

    inline typename gls::image<T>::unique_ptr toImage() const {
        auto image = std::make_unique<gls::image<T>>(gls::image<T>::width, gls::image<T>::height * depth);
        copyPixelsTo(image.get());
        return image;
    }

    void copyPixelsFrom(const image<T>& other) const {
        assert(other.width == image<T>::width && other.height == image<T>::height * depth);
        cl::enqueueWriteImage(_image, true, {0, 0, 0}, {(size_t)image<T>::width, (size_t)image<T>::height, static_cast<size_t>(depth)},
                              image<T>::pixel_size * image<T>::width,
                              image<T>::width * image<T>::height * image<T>::pixel_size, other.pixels().data());
    }

    void copyPixelsTo(image<T>* other) const {
        assert(other->width == image<T>::width && other->height == image<T>::height * depth);
        cl::enqueueReadImage(_image, true, {0, 0, 0}, {(size_t)image<T>::width, (size_t)image<T>::height, static_cast<size_t>(depth)},
                             image<T>::pixel_size * image<T>::width,
                             image<T>::width * image<T>::height * image<T>::pixel_size, other->pixels().data());
    }

    static inline cl::Image2DArray buildImage(cl::Context context, int width, int height, int depth) {
        return cl::Image2DArray(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cl_image<T>::ImageFormat(), depth, width, height, width, width * height);
    }

    cl::Image2DArray getImage2DArray() const { return _image; }
};

template <typename T>
class cl_image_3d : public cl_image<T> {
    const cl::Image3D _image;

   public:
    typedef std::unique_ptr<cl_image_3d<T>> unique_ptr;

    cl_image_3d(cl::Context context, int _width, int _height)
        : cl_image<T>(_width, _height), _image(buildImage(context, _width, _height)) {}

    cl_image_3d(cl::Context context, const gls::image<T>& other) : cl_image_3d(context, other.width, other.height) {
        copyPixelsFrom(other);
    }

    inline typename gls::image<T>::unique_ptr toImage() const {
        auto image = std::make_unique<gls::image<T>>(gls::image<T>::width, gls::image<T>::height);
        copyPixelsTo(image.get());
        return image;
    }

    void copyPixelsFrom(const image<T>& other) const {
        assert(other.width == image<T>::width && other.height == image<T>::height);
        size_t depth = image<T>::height / image<T>::width;
        cl::enqueueWriteImage(_image, true, {0, 0, 0}, {(size_t)image<T>::width, (size_t)image<T>::width, depth},
                              image<T>::pixel_size * image<T>::width,
                              image<T>::width * image<T>::width * image<T>::pixel_size, other.pixels().data());
    }

    void copyPixelsTo(image<T>* other) const {
        assert(other->width == image<T>::width && other->height == image<T>::height);
        size_t depth = image<T>::height / image<T>::width;
        cl::enqueueReadImage(_image, true, {0, 0, 0}, {(size_t)image<T>::width, (size_t)image<T>::width, depth},
                             image<T>::pixel_size * image<T>::width,
                             image<T>::width * image<T>::width * image<T>::pixel_size, other->pixels().data());
    }

    static inline cl::Image3D buildImage(cl::Context context, int width, int height) {
        size_t depth = height / width;
        return cl::Image3D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cl_image<T>::ImageFormat(), width, width, depth);
    }

    cl::Image3D getImage3D() const { return _image; }
};

}  // namespace gls

#endif /* CL_IMAGE_H */
