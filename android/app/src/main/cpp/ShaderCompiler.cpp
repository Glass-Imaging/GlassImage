/*******************************************************************************
 * Copyright (c) 2021-2022 Glass Imaging Inc.
 * Author: Fabio Riccardi <fabio@glass-imaging.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include <fstream>
#include <sstream>

#include "gls_cl.hpp"
#include "gls_logging.h"

static const char* TAG = "ShaderCompiler";

extern "C" int __android_log_print(int prio, const char* tag, const char* fmt, ...) { return 0; }

int main(int argc, const char* argv[]) {
    if (argc > 1 && strcmp(argv[1], "-help") == 0) {
        std::cout << "OpenCL Shader Compiler." << std::endl;
        std::cout << "Usage: cat shader.cl | " << argv[0] << " outfile" << std::endl;
        return 0;
    }

//    // Read shader source from stdin
//    std::stringbuf buffer;
//    std::ostream os(&buffer);
//    for (std::string line; std::getline(std::cin, line);) {
//        os << line << std::endl;
//    }

    std::ifstream t("file.txt");
    std::stringstream buffer;
    buffer << std::cin.rdbuf();

    try {
        gls::OpenCLContext glsContext(/*shadersRootPath=*/"", /*quiet=*/true);
        auto context = glsContext.clContext();

        cl::Program program(buffer.str());
        if (gls::OpenCLContext::buildProgram(program) != 0) {
            return -1;
        }

        std::vector<std::vector<unsigned char>> binaries;
        cl_int result = program.getInfo(CL_PROGRAM_BINARIES, &binaries);
        if (result != CL_SUCCESS) {
            LOG_ERROR(TAG) << "CL_PROGRAM_BINARIES returned: " << gls::clStatusToString(result) << std::endl;
            return -1;
        }

        gls::OpenCLContext::saveBinaryFile(argc > 1 ? argv[1] : "binaryShader.o", binaries[0]);

        return 0;
    } catch (cl::Error& err) {
        LOG_ERROR(TAG) << "Caught Exception: " << err.what() << " - " << gls::clStatusToString(err.err()) << std::endl;
        return -1;
    }
}
