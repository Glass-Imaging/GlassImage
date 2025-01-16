//
//  gls_gpu_image.hpp
//  GlassCamera
//
//  Created by Fabio Riccardi on 7/22/23.
//

#ifndef gls_gpu_image_h
#define gls_gpu_image_h

#include <exception>
#include <map>

#include "gls_image.hpp"

template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;

namespace gls {

class platform_buffer;

class buffer {
   public:
    virtual ~buffer() {}

    virtual const platform_buffer* operator()() const = 0;
};

class platform_buffer : public buffer {
   public:
    virtual size_t bufferSize() const = 0;

    virtual void* mapBuffer() const = 0;

    virtual void unmapBuffer(void* ptr) const = 0;
};

template <typename T>
class gpu_buffer : public buffer {
   protected:
    const std::unique_ptr<platform_buffer> _buffer;

    struct mapped_span : public std::span<T> {
       private:
        std::function<void(void* ptr)> _cleanup;

       public:
        mapped_span(T* data, size_t size, std::function<void(void* ptr)> cleanup)
            : std::span<T>(data, size), _cleanup(cleanup) {}

        virtual ~mapped_span() { _cleanup(this->data()); }
    };

   public:
    gpu_buffer(platform_buffer* buffer) : _buffer(buffer) {}

    virtual const platform_buffer* operator()() const { return _buffer.get(); }

    typedef std::unique_ptr<gpu_buffer<T>> unique_ptr;

    size_t size() const { return _buffer->bufferSize() / sizeof(T); }

    // Returns a self unmapping span, data is guaranteed to be mapped within the span's lifetime
    std::span<T> contents() const {
        return mapped_span((T*)_buffer->mapBuffer(), _buffer->bufferSize() / sizeof(T),
                           [this](void* ptr) { _buffer->unmapBuffer(ptr); });
    }

    // convenience data setter s to avoid mapping/unmapping the buffer by hand
    void setData(const T& val) const {
        auto buffer_contents = contents();
        memcpy(buffer_contents.data(), &val, sizeof(T));
    }

    void setData(const std::span<T>& data) const {
        auto buffer_contents = contents();
        std::copy(data.begin(), data.end(), buffer_contents.data());
    }
};

class platform_texture;

class texture {
   public:
    enum channel_type {
        UNSIGNED_INT8,
        UNSIGNED_INT16,
        UNORM_INT8,
        UNORM_INT16,
        UNSIGNED_INT32,
        SNORM_INT8,
        SNORM_INT16,
        SIGNED_INT32,
        FLOAT32,
        FLOAT16
    };

    struct format {
        const int channels;
        const channel_type dataType;

        int elementSize() {
            int type_size;
            switch (dataType) {
                case UNORM_INT8:
                case SNORM_INT8:
                case UNSIGNED_INT8:
                    type_size = 1;
                    break;
                case UNORM_INT16:
                case SNORM_INT16:
                case UNSIGNED_INT16:
                case FLOAT16:
                    type_size = 2;
                    break;
                case UNSIGNED_INT32:
                case SIGNED_INT32:
                case FLOAT32:
                    type_size = 4;
                    break;
            }
            return type_size * channels;
        }

        format(int _channels, channel_type _dataType) : channels(_channels), dataType(_dataType) {}
    };

    template <typename T>
    static inline format TextureFormat();

    virtual const platform_texture* operator()() const = 0;

    virtual ~texture() {}
};

class platform_texture : public texture {
   public:
    virtual int texture_width() const = 0;

    virtual int texture_height() const = 0;

    virtual int texture_stride() const = 0;

    virtual int pixelSize() const = 0;

    virtual std::span<uint8_t> mapTexture() const = 0;

    virtual void unmapTexture(void* ptr) const = 0;
};

template <typename T>
inline texture::format texture::TextureFormat() {
    static_assert(T::channels == 1 || T::channels == 2 || T::channels == 4);
    static_assert(
        std::is_same<typename T::value_type, float>::value ||
#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
        std::is_same<typename T::value_type, gls::float16_t>::value ||
#endif
        std::is_same<typename T::value_type, uint8_t>::value || std::is_same<typename T::value_type, uint16_t>::value ||
        std::is_same<typename T::value_type, uint32_t>::value || std::is_same<typename T::value_type, int8_t>::value ||
        std::is_same<typename T::value_type, int16_t>::value || std::is_same<typename T::value_type, int32_t>::value);

    channel_type type = std::is_same<typename T::value_type, float>::value ? FLOAT32
#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
                        : std::is_same<typename T::value_type, gls::float16_t>::value ? FLOAT16
#endif

#ifdef OPENCL_MAP_UINT_NORMED
                        : std::is_same<typename T::value_type, uint8_t>::value  ? UNORM_INT8
                        : std::is_same<typename T::value_type, uint16_t>::value ? UNORM_INT16
#else
                        : std::is_same<typename T::value_type, uint8_t>::value  ? UNSIGNED_INT8
                        : std::is_same<typename T::value_type, uint16_t>::value ? UNSIGNED_INT16
#endif
                        : std::is_same<typename T::value_type, uint32_t>::value ? UNSIGNED_INT32
                        : std::is_same<typename T::value_type, int8_t>::value   ? SNORM_INT8
                        : std::is_same<typename T::value_type, int16_t>::value  ? SNORM_INT16
                                                                                : SIGNED_INT32;

    return format{T::channels, type};
}

#define DECLARE_TYPE_FORMATS(data_type, channel_type)                           \
    template <>                                                                 \
    inline texture::format texture::TextureFormat<data_type>() {                \
        return texture::format(1, channel_type);                                \
    }                                                                           \
                                                                                \
    template <>                                                                 \
    inline texture::format texture::TextureFormat<std::array<data_type, 2>>() { \
        return texture::format(2, channel_type);                                \
    }                                                                           \
                                                                                \
    template <>                                                                 \
    inline texture::format texture::TextureFormat<std::array<data_type, 4>>() { \
        return texture::format(4, channel_type);                                \
    }

DECLARE_TYPE_FORMATS(float, FLOAT32)

#if USE_FP16_FLOATS && !(__APPLE__ && __x86_64__)
DECLARE_TYPE_FORMATS(float16_t, FLOAT16)
#endif

#ifdef OPENCL_MAP_UINT_NORMED
DECLARE_TYPE_FORMATS(uint8_t, UNORM_INT8)
DECLARE_TYPE_FORMATS(uint16_t, UNORM_INT16)
#else
DECLARE_TYPE_FORMATS(uint8_t, UNSIGNED_INT8)
DECLARE_TYPE_FORMATS(uint16_t, UNSIGNED_INT16)
#endif
DECLARE_TYPE_FORMATS(uint32_t, UNSIGNED_INT32)

DECLARE_TYPE_FORMATS(int8_t, SNORM_INT8)
DECLARE_TYPE_FORMATS(int16_t, SNORM_INT16)
DECLARE_TYPE_FORMATS(int32_t, SIGNED_INT32)

#undef DECLARE_TYPE_FORMATS

template <typename T>
struct mapped_image : public image<T> {
    std::function<void(void* ptr)> _cleanup;

    mapped_image(int _width, int _height, int _stride, std::span<T> data, std::function<void(void* ptr)> cleanup)
        : image<T>(_width, _height, _stride, data), _cleanup(cleanup) {}

    ~mapped_image() { _cleanup((*this)[0]); }
};

template <typename T>
class gpu_image : public basic_image<T>, public virtual texture {
   protected:
    const std::unique_ptr<platform_texture> _texture;

   public:
    gpu_image(int _width, int _height, platform_texture* texture)
        : basic_image<T>(_width, _height), _texture(texture) {}

    typedef std::unique_ptr<gpu_image<T>> unique_ptr;

    typename gls::image<T>::unique_ptr toImage() const {
        auto image = std::make_unique<gls::image<T>>(gls::image<T>::width, gls::image<T>::height);
        copyPixelsTo(image.get());
        return image;
    }

    void copyPixelsFrom(const image<T>& other) const {
        assert(other.width == image<T>::width && other.height == image<T>::height);
        auto cpuImage = mapImage();
        copyPixels(&cpuImage, other);
    }

    void copyPixelsTo(image<T>* other) const {
        assert(other->width == image<T>::width && other->height == image<T>::height);
        auto cpuImage = mapImage();
        copyPixels(other, cpuImage);
    }

    void apply(std::function<void(T* pixel, int x, int y)> process) {
        auto cpu_image = mapImage();
        for (int y = 0; y < basic_image<T>::height; y++) {
            for (int x = 0; x < basic_image<T>::width; x++) {
                process(&(*cpu_image)[y][x], x, y);
            }
        }
    }

    typename gls::image<T> mapImage() const {
        auto mappedTexture = _texture->mapTexture();

        return mapped_image<T>(basic_image<T>::width, basic_image<T>::height, _texture->texture_stride(),
                               std::span<T>((T*)mappedTexture.data(), mappedTexture.size() / sizeof(T)),
                               [this](void* ptr) { _texture->unmapTexture(ptr); });
    }

    virtual const platform_texture* operator()() const override { return _texture.get(); }
};

class GpuCommandEncoder {
   public:
    virtual ~GpuCommandEncoder() {}

    virtual void setBytes(const void* parameter, size_t parameter_size, unsigned index) = 0;

    virtual void setBuffer(const gls::buffer& buffer, unsigned index) = 0;

    virtual void setTexture(const gls::texture& texture, unsigned index) = 0;
};

class GpuContext {
   public:
    virtual ~GpuContext() {}

    template <typename T>
    typename gpu_image<T>::unique_ptr new_gpu_image_2d(int _width, int _height) {
        auto t = new_platform_texture(_width, _height, texture::TextureFormat<T>());
        return std::make_unique<gpu_image<T>>(_width, _height, t);
    }

    template <typename T>
    typename gpu_image<T>::unique_ptr new_gpu_image_2d(const gls::size& s) {
        return new_gpu_image_2d<T>(s.width, s.height);
    }

    template <typename T>
    typename gpu_image<T>::unique_ptr new_gpu_image_2d(const gls::image<T>& other) {
        auto newImage = new_gpu_image_2d<T>(other.width, other.height);
        newImage->copyPixelsFrom(other);
        return newImage;
    }

    template <typename T>
    typename gpu_buffer<T>::unique_ptr new_buffer(size_t size, bool readOnly = false) {
        auto b = new_platform_buffer(sizeof(T) * size, readOnly);
        return std::make_unique<gpu_buffer<T>>(b);
    }

    template <typename T>
    typename gpu_buffer<T>::unique_ptr new_buffer(const T& val, bool readOnly = false) {
        auto b = std::make_unique<gpu_buffer<T>>(new_platform_buffer(sizeof(T), readOnly));
        b->setData(val);
        return b;
    }

    template <typename T>
    typename gpu_buffer<T>::unique_ptr new_buffer(const std::vector<T>& data, bool readOnly = false) {
        auto b = std::make_unique<gpu_buffer<T>>(new_platform_buffer(sizeof(T) * data.size(), readOnly));
        b->setData(std::span<T>((T*)data.data(), (size_t)data.size()));
        return b;
    }

    template <typename T>
    typename gpu_buffer<T>::unique_ptr new_buffer(const std::span<T>& data, bool readOnly = false) {
        auto b = std::make_unique<gpu_buffer<T>>(new_platform_buffer(sizeof(T) * data.size(), readOnly));
        b->setData(data);
        return b;
    }

    virtual platform_buffer* new_platform_buffer(size_t size, bool readOnly) = 0;

    virtual platform_texture* new_platform_texture(int _width, int _height, texture::format format) = 0;

    virtual void enqueue(const std::string& kernelName, const gls::size& gridSize, const gls::size& threadGroupSize,
                         std::function<void(GpuCommandEncoder*)> encodeKernelParameters,
                         std::function<void(void)> completionHandler) = 0;

    virtual void enqueue(const std::string& kernelName, const gls::size& gridSize,
                         std::function<void(GpuCommandEncoder*)> encodeKernelParameters,
                         std::function<void(void)> completionHandler) = 0;

    void enqueue(const std::string& kernelName, const gls::size& gridSize,
                 std::function<void(GpuCommandEncoder*)> encodeKernelParameters) {
        enqueue(kernelName, gridSize, encodeKernelParameters, []() {});
    }

    void enqueue(const std::string& kernelName, const gls::size& gridSize, const gls::size& threadGroupSize,
                 std::function<void(GpuCommandEncoder*)> encodeKernelParameters) {
        enqueue(kernelName, gridSize, threadGroupSize, encodeKernelParameters, []() {});
    }

    virtual void waitForCompletion() = 0;
};

template <typename... Ts>
class Kernel {
    const std::string _kernelName;

    template <int index, typename T0, typename... T1s>
    void setArgs(GpuCommandEncoder* encoder, const T0&& t0, const T1s&&... t1s) const {
        setParameter(encoder, t0, index);
        setArgs<index + 1, const T1s...>(encoder, std::forward<const T1s>(t1s)...);
    }

    template <int index, typename T0>
    void setArgs(GpuCommandEncoder* encoder, const T0&& t0) const {
        setParameter(encoder, t0, index);
    }

   public:
    Kernel(GpuContext* context, const std::string& kernelName) : _kernelName(kernelName) {
        // auto pipelineState = context->getPipelineState(kernelName);
    }

    ~Kernel() {}

    // TODO: should we implement some sort of setVector method for encoder?
    template <size_t N, typename T>
    void setParameter(GpuCommandEncoder* encoder, const gls::Vector<N, T>& vec, unsigned index) const {
        encoder->setBytes(vec.data(), sizeof(T) * vec.size(), index);
    }

    template <Derived<gls::buffer> T>
    void setParameter(GpuCommandEncoder* encoder, const T& val, unsigned index) const {
        const gls::buffer* buffer = dynamic_cast<const gls::buffer*>(&val);
        encoder->setBuffer(*buffer, index);
    }

    template <Derived<gls::texture> T>
    void setParameter(GpuCommandEncoder* encoder, const T& val, unsigned index) const {
        const gls::texture* texture = dynamic_cast<const gls::texture*>(&val);
        encoder->setTexture(*texture, index);
    }

    template <typename T>
    void setParameter(GpuCommandEncoder* encoder, const T& val, unsigned index) const {
        encoder->setBytes(&val, sizeof(T), index);
    }

    // All kernel parameters are passed by const reference
    void operator()(GpuContext* context, const gls::size& gridSize, const gls::size& threadGroupSize,
                    const Ts&... ts) const {
        context->enqueue(_kernelName, gridSize, threadGroupSize,
                         [&, this](GpuCommandEncoder* encoder) { setArgs<0>(encoder, std::forward<const Ts>(ts)...); });  
    }

    void operator()(GpuContext* context, const gls::size& gridSize, const Ts&... ts) const {
        context->enqueue(_kernelName, gridSize,
                         [&, this](GpuCommandEncoder* encoder) { setArgs<0>(encoder, std::forward<const Ts>(ts)...); });
    }
};
}  // namespace gls

#endif /* gls_gpu_image_h */
