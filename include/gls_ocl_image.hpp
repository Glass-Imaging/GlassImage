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

#include "gls_cl.hpp"
#include "gls_gpu_image.hpp"
#include "gls_image.hpp"

namespace gls {

class ocl_texture : public virtual platform_texture {
   protected:
    cl::Buffer _buffer;
    cl::Image2D _image;

   public:
    cl::Image2D image() const { return _image; }
    cl::Buffer buffer() const { return _buffer; }

    int texture_width() const override { return (int)_image.getImageInfo<CL_IMAGE_WIDTH>(); }

    int texture_height() const override { return (int)_image.getImageInfo<CL_IMAGE_HEIGHT>(); }

    int texture_stride() const override {
        return (int)_image.getImageInfo<CL_IMAGE_ROW_PITCH>() / _image.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();
    }

    int pixelSize() const override { return (int)_image.getImageInfo<CL_IMAGE_ELEMENT_SIZE>(); }

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

#ifdef OPENCL_MAP_UINT_NORMED
                               : format.dataType == UNORM_INT8  ? CL_UNORM_INT8
                               : format.dataType == UNORM_INT16 ? CL_UNORM_INT16

#else
                               : format.dataType == UNSIGNED_INT8  ? CL_UNSIGNED_INT8
                               : format.dataType == UNSIGNED_INT16 ? CL_UNSIGNED_INT16
#endif
                               : format.dataType == UNSIGNED_INT32 ? CL_UNSIGNED_INT32
                               : format.dataType == SNORM_INT8     ? CL_SNORM_INT8
                               : format.dataType == SNORM_INT16    ? CL_SNORM_INT16
                                                                   : CL_SIGNED_INT32;

        return cl::ImageFormat(order, type);
    }

    ocl_texture(cl::Context context, int _width, int _height, format textureFormat) {
        const int pitch_alignment = cl::Device::getDefault().getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();
        const int stride = pitch_alignment * ((_width + pitch_alignment - 1) / pitch_alignment);
        const int element_size = textureFormat.elementSize();
        _buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, stride * _height * element_size);
        _image = cl::Image2D(context, imageFormat(textureFormat), _buffer, _width, _height, stride * element_size);
    }

    std::span<uint8_t> mapTexture() const override {
        cl::CommandQueue queue = cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        cl::Event map_event;
        size_t buffer_size = pixelSize() * texture_stride() * texture_height();
        void* mapped_data = queue.enqueueMapBuffer(_buffer, false, CL_MAP_READ | CL_MAP_WRITE, 0, buffer_size, nullptr, &map_event);
        map_event.wait();
        return std::span<uint8_t>((uint8_t*)mapped_data, buffer_size);
    }

    void unmapTexture(void* ptr) const override {
        cl::CommandQueue queue = cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        queue.enqueueUnmapMemObject(ocl_texture::_buffer, ptr);
    }

    virtual const class platform_texture* operator()() const override { return this; }
};

class ocl_buffer : public platform_buffer {
    const cl::Buffer _buffer;

   public:
    // Create new buffer
    ocl_buffer(cl::Context context, size_t length, bool readOnly)
        : _buffer(cl::Buffer(context, readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE, length)) {}
    
    // Wrap existing buffer (e.g., from QNN ION buffer)
    ocl_buffer(const cl::Buffer& existingBuffer) : _buffer(existingBuffer) {}

    cl::Buffer buffer() const { return _buffer; }

    virtual size_t bufferSize() const override { return (int)_buffer.getInfo<CL_MEM_SIZE>(); }

    virtual void* mapBuffer() const override {
        // Use out-of-order queue to avoid blocking the default queue
        cl::CommandQueue queue = cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        cl::Event map_event;
        void* mapped_data = queue.enqueueMapBuffer(_buffer, false, CL_MAP_READ | CL_MAP_WRITE, 0, bufferSize(), nullptr, &map_event);
        map_event.wait();
        return mapped_data;
    }

    virtual void unmapBuffer(void* ptr) const override {
        // Use out-of-order queue to avoid blocking the default queue
        cl::CommandQueue queue = cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        queue.enqueueUnmapMemObject(_buffer, ptr);
    }

    virtual const class platform_buffer* operator()() const override { return this; }
};

}  // namespace gls

#endif /* gls_ocl_image_h */
