#include "gls_image.hpp"
#include "gls_logging.h"
#include "gls_ocl.hpp"

int main() {
    gls::image<gls::rgb_pixel> image(512, 512);
    gls::logging::current_log_level = gls::logging::LOG_LEVEL_INFO;
    gls::logging::LogInfo("GlassImageTest") << "Image created: " << image.width << "x" << image.height << std::endl;
}