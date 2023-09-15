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

#include "gls_mtl_image.hpp"

namespace gls {

MTL::PixelFormat mtl_texture::pixelFormat(const texture::format& format) {
    if (format.channels != 1 && format.channels != 2 && format.channels != 4) {
        throw std::runtime_error("Unexpected Texture Channels Count");
    }
    switch (format.dataType) {
        case UNORM_INT8:
            return format.channels == 1   ? MTL::PixelFormatR8Unorm
                   : format.channels == 2 ? MTL::PixelFormatRG8Unorm
                                          : MTL::PixelFormatRGBA8Unorm;

        case UNORM_INT16:
            return format.channels == 1   ? MTL::PixelFormatR16Unorm
                   : format.channels == 2 ? MTL::PixelFormatRG16Unorm
                                          : MTL::PixelFormatRGBA16Unorm;

        case UNSIGNED_INT32:
            return format.channels == 1   ? MTL::PixelFormatR32Uint
                   : format.channels == 2 ? MTL::PixelFormatRG32Uint
                                          : MTL::PixelFormatRGBA32Uint;

        case SNORM_INT8:
            return format.channels == 1   ? MTL::PixelFormatR8Snorm
                   : format.channels == 2 ? MTL::PixelFormatRG8Snorm
                                          : MTL::PixelFormatRGBA8Snorm;

        case SNORM_INT16:
            return format.channels == 1   ? MTL::PixelFormatR16Snorm
                   : format.channels == 2 ? MTL::PixelFormatRG16Snorm
                                          : MTL::PixelFormatRGBA16Snorm;

        case SIGNED_INT32:
            return format.channels == 1   ? MTL::PixelFormatR32Sint
                   : format.channels == 2 ? MTL::PixelFormatRG32Sint
                                          : MTL::PixelFormatRGBA32Sint;

        case FLOAT32:
            return format.channels == 1   ? MTL::PixelFormatR32Float
                   : format.channels == 2 ? MTL::PixelFormatRG32Float
                                          : MTL::PixelFormatRGBA32Float;

        case FLOAT16:
            return format.channels == 1   ? MTL::PixelFormatR16Float
                   : format.channels == 2 ? MTL::PixelFormatRG16Float
                                          : MTL::PixelFormatRGBA16Float;

        default:
            throw std::runtime_error("Unexpected Texture Data Type");
    }
}

int mtl_texture::pixelSize(MTL::PixelFormat format) {
    switch(format) {
        case MTL::PixelFormatR8Unorm:           return 1 * sizeof(uint8_t);
        case MTL::PixelFormatRG8Unorm:          return 2 * sizeof(uint8_t);
        case MTL::PixelFormatRGBA8Unorm:        return 4 * sizeof(uint8_t);

        case MTL::PixelFormatR16Unorm:          return 1 * sizeof(uint16_t);
        case MTL::PixelFormatRG16Unorm:         return 2 * sizeof(uint16_t);
        case MTL::PixelFormatRGBA16Unorm:       return 4 * sizeof(uint16_t);

        case MTL::PixelFormatR32Uint:           return 1 * sizeof(uint32_t);
        case MTL::PixelFormatRG32Uint:          return 2 * sizeof(uint32_t);
        case MTL::PixelFormatRGBA32Uint:        return 4 * sizeof(uint32_t);

        case MTL::PixelFormatR8Snorm:           return 1 * sizeof(int8_t);
        case MTL::PixelFormatRG8Snorm:          return 2 * sizeof(int8_t);
        case MTL::PixelFormatRGBA8Snorm:        return 4 * sizeof(int8_t);

        case MTL::PixelFormatR16Snorm:          return 1 * sizeof(int16_t);
        case MTL::PixelFormatRG16Snorm:         return 2 * sizeof(int16_t);
        case MTL::PixelFormatRGBA16Snorm:       return 4 * sizeof(int16_t);

        case MTL::PixelFormatR32Sint:           return 1 * sizeof(int32_t);
        case MTL::PixelFormatRG32Sint:          return 2 * sizeof(int32_t);
        case MTL::PixelFormatRGBA32Sint:        return 4 * sizeof(int32_t);

        case MTL::PixelFormatR32Float:          return 1 * sizeof(float);
        case MTL::PixelFormatRG32Float:         return 2 * sizeof(float);
        case MTL::PixelFormatRGBA32Float:       return 4 * sizeof(float);

        case MTL::PixelFormatR16Float:          return 1 * sizeof(float16_t);
        case MTL::PixelFormatRG16Float:         return 2 * sizeof(float16_t);
        case MTL::PixelFormatRGBA16Float:       return 4 * sizeof(float16_t);

        default:
            throw std::runtime_error("Unexpected PixelFormat");
    }
}

uint32_t mtl_texture::computeStride(MTL::Device* device, MTL::PixelFormat pixelFormat, int _width) {
    int pixelByteSize = pixelSize(pixelFormat);
    const uint32_t mlta = (uint32_t)device->minimumLinearTextureAlignmentForPixelFormat(pixelFormat);
    uint32_t bytesPerRow = mlta * ((pixelByteSize * _width + mlta - 1) / mlta);
    return bytesPerRow / pixelByteSize;
}

mtl_texture::mtl_texture(MTL::Device* device, int _width, int _height, mtl_texture::format textureFormat) {
    const auto format = pixelFormat(textureFormat);
    const int mlta = (int) device->minimumLinearTextureAlignmentForPixelFormat(format);
    int bytesPerRow = mlta * ((pixelSize(format) * _width + mlta - 1) / mlta);

    auto textureDesc = MTL::TextureDescriptor::texture2DDescriptor(format, _width, _height, /*mipmapped=*/false);
    textureDesc->setStorageMode(MTL::StorageModeShared);
    textureDesc->setUsage(MTL::ResourceUsageSample | MTL::ResourceUsageRead | MTL::ResourceUsageWrite);

    _buffer = NS::TransferPtr(device->newBuffer(bytesPerRow * _height, MTL::ResourceStorageModeShared));
    _texture = NS::TransferPtr(_buffer->newTexture(textureDesc, 0, bytesPerRow));
}

}  // namespace gls
