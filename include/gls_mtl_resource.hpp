#pragma once

#include <memory>
#include <CoreFoundation/CoreFoundation.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace gls {

#ifdef __OBJC__
// Helper for bridging between Obj-C and C++ Metal types
template<typename T>
struct mtl_bridge;

template<>
struct mtl_bridge<MTL::Buffer> {
    using objc_type = id<MTLBuffer>;
};
#endif

// RAII wrapper for CoreFoundation/Metal resources
template<typename T>
class mtl_resource {
private:
    T* resource;

public:
    mtl_resource() : resource(nullptr) {}
    
    explicit mtl_resource(T* ptr) : resource(ptr) {}
    
    // Take ownership via bridging
    #ifdef __OBJC__
    static mtl_resource bridging_retain(typename mtl_bridge<T>::objc_type obj) {
        return mtl_resource((T*)CFBridgingRetain(obj));
    }
    #endif
    
    ~mtl_resource() {
        if (resource) {
            #ifdef __OBJC__
            CFRelease(resource);
            #endif
            resource = nullptr;
        }
    }
    
    // Copy constructor and assignment
    mtl_resource(const mtl_resource& other) {
        resource = other.resource;
        if (resource) {
            #ifdef __OBJC__
            CFRetain(resource);
            #endif
        }
    }
    
    mtl_resource& operator=(const mtl_resource& other) {
        if (this != &other) {
            if (resource) {
                #ifdef __OBJC__
                CFRelease(resource);
                #endif
            }
            resource = other.resource;
            if (resource) {
                #ifdef __OBJC__
                CFRetain(resource);
                #endif
            }
        }
        return *this;
    }
    
    // Move semantics
    mtl_resource(mtl_resource&& other) noexcept : resource(other.resource) {
        other.resource = nullptr;
    }
    
    mtl_resource& operator=(mtl_resource&& other) noexcept {
        if (this != &other) {
            if (resource) {
                #ifdef __OBJC__
                CFRelease(resource);
                #endif
            }
            resource = other.resource;
            other.resource = nullptr;
        }
        return *this;
    }
    
    // Access
    T* get() const { return resource; }
    operator bool() const { return resource != nullptr; }
    
    // Release ownership
    T* release() {
        T* tmp = resource;
        resource = nullptr;
        return tmp;
    }
};

namespace metal {
    // Forward declare Metal types when not in Objective-C context
    #ifndef __OBJC__
    namespace MTL { class Buffer; }
    #endif

    // Specialization for MTLBuffer
    using buffer = mtl_resource<MTL::Buffer>;
}

} // namespace gls 