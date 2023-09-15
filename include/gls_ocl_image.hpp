//
//  gls_ocl_image.hpp
//  GlassCamera
//
//  Created by Fabio Riccardi on 8/7/23.
//

#ifndef gls_ocl_image_h
#define gls_ocl_image_h

#include <exception>
#include <map>

#include "gls_image.hpp"
#include "gls_gpu_image.hpp"
#include "gls_cl.hpp"

namespace gls {

class ocl_texture : public virtual platform_texture {
protected:
    cl::Image2D _image;

public:
    cl::Image2D image() const {
        return _image;
    }

    int texture_width() const override {
        return (int) _image.getImageInfo<CL_IMAGE_WIDTH>();
    }

    int texture_height() const override {
        return (int) _image.getImageInfo<CL_IMAGE_HEIGHT>();
    }

    int texture_stride() const override {
        return (int) _image.getImageInfo<CL_IMAGE_ROW_PITCH>() / _image.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();
    }

    int pixelSize() const override {
        return (int) _image.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();
    }

    static int pixelSize(const cl::ImageFormat& format);

    static cl::ImageFormat imageFormat(const texture::format& format) {
        if (format.channels != 1 && format.channels != 2 && format.channels != 4) {
            throw std::runtime_error("Unexpected Texture Channels Count");
        }

        cl_channel_order order = format.channels == 1 ? CL_R : format.channels == 2 ? CL_RG : CL_RGBA;
        cl_channel_type type = format.dataType == FLOAT32 ? CL_FLOAT
    #if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
                               : format.dataType == FLOAT16 ? CL_HALF_FLOAT
    #endif
                               : format.dataType == UNORM_INT8     ? CL_UNORM_INT8
                               : format.dataType == UNORM_INT16    ? CL_UNORM_INT16
                               : format.dataType == UNSIGNED_INT32 ? CL_UNSIGNED_INT32
                               : format.dataType == SNORM_INT8     ? CL_SNORM_INT8
                               : format.dataType == SNORM_INT16    ? CL_SNORM_INT16
                                                                   : CL_SIGNED_INT32;

        return cl::ImageFormat(order, type);
    }

    ocl_texture(cl::Context context, int _width, int _height, format textureFormat) :
        _image(cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imageFormat(textureFormat), _width, _height))
        { }

    std::span<uint8_t> mapTexture() const override {
        size_t row_pitch;
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        void* mapped_data = queue.enqueueMapImage(_image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, {0, 0, 0},
                                                  {static_cast<size_t>(texture_width()), static_cast<size_t>(texture_height()), 1}, &row_pitch,
                                                  /*slice_pitch=*/nullptr);
        assert(mapped_data != nullptr);
        assert(texture_stride() == row_pitch / pixelSize());

        size_t data_size = row_pitch * texture_height();

        return std::span<uint8_t>((uint8_t*) mapped_data, data_size);
    }

    void unmapTexture(void* ptr) const override {
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        queue.enqueueUnmapMemObject(ocl_texture::_image, ptr);
    }

    virtual const class platform_texture* operator() () const override {
        return this;
    }
};

class ocl_buffer : public platform_buffer {
    const cl::Buffer _buffer;

public:
    ocl_buffer(cl::Context context, size_t lenght, bool readOnly) :
        // TODO: verify that CL_MEM_ALLOC_HOST_PTR is the best generic access mode
        _buffer(cl::Buffer(context, readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE, lenght)) { }

    cl::Buffer buffer() const {
        return _buffer;
    }

    virtual size_t bufferSize() const override {
        return (int) _buffer.getInfo<CL_MEM_SIZE>();
    }

    virtual void* mapBuffer() const override {
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        return queue.enqueueMapBuffer(_buffer, true, CL_MAP_READ | CL_MAP_WRITE, 0, bufferSize());
    }

    virtual void unmapBuffer(void* ptr) const override {
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        queue.enqueueUnmapMemObject(_buffer, ptr);
    }

    virtual const class platform_buffer* operator() () const override {
        return this;
    }
};

}  // namespace gls

#endif /* gls_ocl_image_h */
