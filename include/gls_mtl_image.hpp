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

#ifndef gls_mtl_image_h
#define gls_mtl_image_h

#include <exception>
#include <map>

#include "gls_image.hpp"
#include "gls_gpu_image.hpp"

#include <Metal/Metal.hpp>

namespace gls {

class mtl_texture : public virtual platform_texture {
protected:
    NS::SharedPtr<MTL::Buffer> _buffer;
    NS::SharedPtr<MTL::Texture> _texture;

public:
    int texture_width() const override {
        return (int) _texture->width();
    }

    int texture_height() const override {
        return (int) _texture->height();
    }

    int texture_stride() const override {
        int pixelByteSize = pixelSize(_texture->pixelFormat());
        return (int) _texture->bufferBytesPerRow() / pixelByteSize;
    }

    int pixelSize() const override {
        return pixelSize(_texture->pixelFormat());
    }

    static MTL::PixelFormat pixelFormat(const texture::format& format);

    static int pixelSize(MTL::PixelFormat format);

    static uint32_t computeStride(MTL::Device* device, MTL::PixelFormat pixelFormat, int _width);

    // Constructor for creating from scratch
    mtl_texture(MTL::Device* device, int _width, int _height, format pixelFormat);

    // Constructor for wrapping existing texture
    // TODO: Test. mtl_buffer was tested, this one was not.
    mtl_texture(MTL::Texture* existingTexture) : _texture(NS::RetainPtr(existingTexture)) {
        // If the texture is backed by a buffer, get it
        if (existingTexture->buffer()) {
            _buffer = NS::RetainPtr(existingTexture->buffer());
        }
    }

    const MTL::Buffer* buffer() const { return _buffer.get(); }

    const MTL::Texture* texture() const { return _texture.get(); }

    std::span<uint8_t> mapTexture() const override {
        uint8_t* bufferData = (uint8_t*) _buffer->contents();
        size_t bufferLength = _buffer->length();
        return std::span(bufferData, bufferLength);
    }

    void unmapTexture(void* ptr) const override { }

    virtual const class platform_texture* operator() () const override {
        return this;
    }
};

struct mtl_buffer : public platform_buffer {
    const NS::SharedPtr<MTL::Buffer> _buffer;

    // Constructor for creating from scratch
    mtl_buffer(MTL::Device* device, size_t lenght) :
        _buffer(NS::TransferPtr(device->newBuffer(lenght, MTL::ResourceStorageModeShared))) { }

    // Constructor for wrapping existing buffer
    mtl_buffer(MTL::Buffer* existingBuffer) : _buffer(NS::RetainPtr(existingBuffer)) { }

    virtual size_t bufferSize() const override {
        return _buffer->length();
    }

    virtual void* mapBuffer() const override {
        return _buffer->contents();
    }

    virtual void unmapBuffer(void* ptr) const override { }

    virtual const class platform_buffer* operator() () const override {
        return this;
    }
};

}  // namespace gls

#endif /* gls_mtl_image_h */
