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

#ifndef auto_ptr_h
#define auto_ptr_h

namespace gls {

// exception friendly pointer with destructor
template <typename T>
struct auto_ptr : std::unique_ptr<T, std::function<void(T*)>> {
   public:
    auto_ptr(T* val, std::function<void(T*)> destroyer) : std::unique_ptr<T, std::function<void(T*)>>(val, destroyer) {}

    // Allow the auto_ptr to be implicitly converted to the underlying pointer type
    operator T*() const { return this->get(); }
    operator T*() { return this->get(); }
};

}  // namespace gls

#endif /* auto_ptr_h */
