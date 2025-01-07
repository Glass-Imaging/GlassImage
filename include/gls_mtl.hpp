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

#ifndef gls_mtl_hpp
#define gls_mtl_hpp

#include <exception>
#include <functional>
#include <map>
#include <vector>
#include <string>
#include <mutex>

#include <Metal/Metal.hpp>

#include "gls_gpu_image.hpp"
#include "gls_mtl_image.hpp"

namespace gls {

// Metal execution context implementing a simple sequential pipeline

class MetalCommandEncoder : public GpuCommandEncoder {
    MTL::ComputeCommandEncoder* _encoder;

public:
    MetalCommandEncoder(MTL::ComputeCommandEncoder* encoder) : _encoder(encoder) { }

    virtual ~MetalCommandEncoder() { }

    virtual void setBytes(const void* parameter, size_t parameter_size, unsigned index) override {
        _encoder->setBytes(parameter, parameter_size, index);
    }

    virtual void setBuffer(const gls::buffer& buffer, unsigned index) override {
        if (const mtl_buffer* b = dynamic_cast<const mtl_buffer*>(buffer())) {
            _encoder->setBuffer(b->_buffer.get(), /*offset=*/ 0, index);
        } else {
            throw std::runtime_error("Unexpected buffer type.");
        }
    }

    virtual void setTexture(const gls::texture& texture, unsigned index) override {
        if (auto t = dynamic_cast<gls::mtl_texture const *>(texture())) {
            _encoder->setTexture(t->texture(), index);
        } else {
            throw std::runtime_error("Unexpected texture type.");
        }
    }
};

class MetalContext : public GpuContext {
    NS::SharedPtr<MTL::Device> _device;
    NS::SharedPtr<MTL::Library> _computeLibrary;
    NS::SharedPtr<MTL::CommandQueue> _commandQueue;
    std::vector<MTL::CommandBuffer*> _work_in_progress;
    std::mutex _work_in_progress_mutex;

    std::unique_ptr<std::map<const std::string, NS::SharedPtr<MTL::ComputePipelineState>>> _kernelStateMap;

public:
    MetalContext(NS::SharedPtr<MTL::Device> device) : _device(device) {
        _computeLibrary = NS::TransferPtr(_device->newDefaultLibrary());
        _commandQueue = NS::TransferPtr(_device->newCommandQueue());
        _kernelStateMap = std::make_unique<std::map<const std::string, NS::SharedPtr<MTL::ComputePipelineState>>>();
    }

    virtual ~MetalContext() {
        waitForCompletion();
    }

    virtual void waitForCompletion() override {
        while (true) {
            MTL::CommandBuffer* commandBuffer = nullptr;
            {
                std::lock_guard<std::mutex> guard(_work_in_progress_mutex);

                if (!_work_in_progress.empty()) {
                    commandBuffer = _work_in_progress[_work_in_progress.size() - 1];
                } else {
                    break;
                }
            }
            if (commandBuffer) {
                commandBuffer->waitUntilCompleted();
            }
        };
    }

    MTL::Device* device() const {
        return _device.get();
    }

    MTL::CommandQueue* commandQueue() const {
        return _commandQueue.get();
    }

    MTL::ComputePipelineState* newKernelPipelineState(const std::string& kernelName) const {
        NS::Error* error = nullptr;
        auto kernel = NS::TransferPtr(_computeLibrary->newFunction(NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding)));
        auto pso = _device->newComputePipelineState(kernel.get(), &error);
        if (!pso) {
            throw std::runtime_error("Couldn't create pipeline state for kernel " + kernelName + " : " + error->localizedDescription()->utf8String());
        }
        return pso;
    }

    NS::SharedPtr<MTL::ComputePipelineState> getPipelineState(const std::string& kernelName) {
        if (auto entry = _kernelStateMap->find(kernelName); entry != _kernelStateMap->end()) {
            return entry->second;
        } else {
            auto newPipelineState = NS::TransferPtr(newKernelPipelineState(kernelName));
            (*_kernelStateMap)[kernelName] = newPipelineState;
            return newPipelineState;
        }
    }

    virtual platform_buffer* new_platform_buffer(size_t size, bool readOnly) override {
        return new mtl_buffer(device(), size);
    }

    virtual platform_texture* new_platform_texture(int _width, int _height, texture::format format) override {
        return new mtl_texture(device(), _width, _height, format);
    }

    virtual void enqueue(const std::string& kernelName, const gls::size& gridSize, const gls::size& threadGroupSize,
                         std::function<void(GpuCommandEncoder*)> encodeKernelParameters,
                         std::function<void(void)> completionHandler) override {
        auto commandBuffer = _commandQueue->commandBuffer();

        // Add commandBuffer to _work_in_progress
        {
            std::lock_guard<std::mutex> guard(_work_in_progress_mutex);
            _work_in_progress.push_back(commandBuffer);
        }

        // Schedule task
        auto encoder = commandBuffer->computeCommandEncoder();

        if (encoder) {
            auto pipelineState = getPipelineState(kernelName);

            encoder->setComputePipelineState(pipelineState.get());

            auto gpuEncoder = MetalCommandEncoder(encoder);
            encodeKernelParameters(&gpuEncoder);

            encoder->dispatchThreads(/*threadsPerGrid=*/ MTL::Size(gridSize.width, gridSize.height, 1),
                                     /*threadsPerThreadgroup*/ MTL::Size(threadGroupSize.width, threadGroupSize.height, 1));
            encoder->endEncoding();
        }

        commandBuffer->addCompletedHandler((MTL::HandlerFunction) [this, completionHandler](MTL::CommandBuffer* commandBuffer) {
            completionHandler();

            // Remove completed commandBuffer from _work_in_progress
            {
                std::lock_guard<std::mutex> guard(_work_in_progress_mutex);
                _work_in_progress.erase(std::remove(_work_in_progress.begin(), _work_in_progress.end(), commandBuffer), _work_in_progress.end());
            }
        });

        commandBuffer->commit();
    }

    virtual void enqueue(const std::string& kernelName, const gls::size& gridSize,
                         std::function<void(GpuCommandEncoder*)> encodeKernelParameters,
                         std::function<void(void)> completionHandler) override {
        auto pipelineState = getPipelineState(kernelName);
        auto threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
        enqueue(kernelName, gridSize, gls::size((int) threadGroupSize, 1), encodeKernelParameters, completionHandler);
    }
};

}  // namespace gls

#endif /* gls_mtl_hpp */
