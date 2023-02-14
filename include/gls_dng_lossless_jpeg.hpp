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

#include <span>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include <stdint.h>
#include <cstring>
#include <stdexcept>
#endif

namespace gls {

class dng_stream {
    const std::span<uint8_t> _buffer;
    size_t _position;

   public:
    dng_stream(uint8_t* buffer, size_t size) : _buffer(std::span(buffer, size)), _position(0) {}

    int8_t Get_uint8() {
        if (_position < _buffer.size()) {
            return _buffer[_position++];
        }
        return -1;
    }

    void Put(const void* data, uint32_t count) {
        if (_position + count < _buffer.size()) {
            std::memcpy(_buffer.data() + _position, data, count);
            _position += count;
        } else {
            throw std::runtime_error("buffer overrun");
        }
    }

    void Skip(size_t delta) {
        size_t newPosition = _position + delta;
        if (newPosition >= 0 && newPosition < _buffer.size()) {
            _position = newPosition;
        } else {
            throw std::runtime_error("buffer overrun");
        }
    }

    void SetReadPosition(size_t offset) {
        size_t newPosition = offset;
        if (newPosition >= 0 && newPosition < _buffer.size()) {
            _position = newPosition;
        } else {
            throw std::runtime_error("buffer overrun");
        }
    }

    size_t Position() const { return _position; }
};

class dng_spooler {
    std::vector<uint8_t> _storage;

   public:
    void Spool(const void* data, uint32_t count) {
        const auto blob = std::span((uint8_t*)data, count);
        _storage.insert(_storage.end(), blob.begin(), blob.end());
    }

    void* data() { return (void*)_storage.data(); }

    size_t size() { return _storage.size(); }
};

void DecodeLosslessJPEG(dng_stream& stream, dng_spooler& spooler, uint32_t minDecodedSize,
                        uint32_t maxDecodedSize, bool bug16, uint64_t endOfData);

void EncodeLosslessJPEG(const uint16_t* srcData, uint32_t srcRows, uint32_t srcCols,
                        uint32_t srcChannels, uint32_t srcBitDepth, int32_t srcRowStep,
                        int32_t srcColStep, dng_stream& stream);

}  // namespace gls
