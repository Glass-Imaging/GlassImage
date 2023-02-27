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

#ifndef GLS_LOGGING_H
#define GLS_LOGGING_H

#include <iostream>

namespace gls {

enum log_level { LOG_LEVEL_ERROR = 0, LOG_LEVEL_INFO = 1, LOG_LEVEL_DEBUG = 2 };

extern log_level currentLogLevel;

class NullStream : public std::ostream {
   public:
    NullStream() : std::ostream(nullptr) {}
    NullStream(const NullStream&) : std::ostream(nullptr) {}
};

extern NullStream null;

inline std::ostream& __log_null(std::ostream& os) { return os; }

}  // namespace gls

#if defined(__ANDROID__) && !defined(USE_IOSTREAM_LOG)

#include <android/log.h>

std::ostream __log_prefix(android_LogPriority level, const std::string& TAG);

#define LOG_INFO(TAG) __log_prefix(ANDROID_LOG_INFO, TAG)
#define LOG_ERROR(TAG) __log_prefix(ANDROID_LOG_ERROR, TAG)
#define LOG_DEBUG(TAG) __log_prefix(ANDROID_LOG_DEBUG, TAG)

#else

std::ostream& __log_prefix(std::ostream& os);

#define LOG_STREAM(label) (gls::currentLogLevel >= label ? __log_prefix(std::cout) : __log_null(gls::null))
#define LOG_INFO(TAG) (LOG_STREAM(gls::LOG_LEVEL_INFO) << "I/" << TAG << ": ")
#define LOG_ERROR(TAG) (LOG_STREAM(gls::LOG_LEVEL_ERROR) << "E/" << TAG << ": ")
#define LOG_DEBUG(TAG) (LOG_STREAM(gls::LOG_LEVEL_DEBUG) << "I/" << TAG << ": ")

#endif

#endif  // GLS_LOGGING_H
