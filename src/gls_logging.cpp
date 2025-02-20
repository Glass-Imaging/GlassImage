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

#include "gls_logging.h"
#include <mutex>

namespace gls {

log_level currentLogLevel = LOG_LEVEL_ERROR;

NullStream null;

}  // namespace gls

#if defined(__ANDROID__) && !defined(USE_IOSTREAM_LOG)

#include <sstream>

struct AndroidLogBuf : public std::streambuf {
    AndroidLogBuf() = default;

    std::streambuf& operator()(android_LogPriority PRIORITY, const std::string TAG) {
        std::lock_guard<std::mutex> lock(_mutex);
        _PRIORITY = PRIORITY;
        _TAG = TAG;
        return *this;
    }

   protected:
    std::streamsize xsputn(const char_type* s, std::streamsize n) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _buf.sputn(s, n);
        return n;
    }

    int_type overflow(int_type ch) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _buf.sputc(ch);
        __android_log_print(_PRIORITY, _TAG.c_str(), "%s", _buf.str().c_str());
        _buf.str("");
        return ch;
    }

   private:
    android_LogPriority _PRIORITY = ANDROID_LOG_INFO;
    std::string _TAG = "Default";
    std::stringbuf _buf;
    std::mutex _mutex;
};

std::ostream __log_prefix(android_LogPriority level, const std::string& TAG) {
    static auto buf = AndroidLogBuf();
    // Prepend Gls to every log tag to make it easier to filter
    return std::ostream(&buf(level, "Gls " + TAG));
}

#else

#include <sys/time.h>

#include <iomanip>
#include <iostream>

std::ostream& __log_prefix(std::ostream& os) {
    timeval time_now;
    gettimeofday(&time_now, nullptr);
    char mbstr[100];
    std::strftime(mbstr, sizeof(mbstr), "%F %T", std::localtime(&time_now.tv_sec));
    return os << mbstr << "." << std::setfill('0') << std::setw(3) << time_now.tv_usec / 1000 << " - ";
}

#endif
