#include "gls_ocl.hpp"

using namespace std;

int main() {
    gls::OCLContext context({});
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_FLOAT;

    cl_image_desc image_desc;
    memset(&image_desc, 0, sizeof(image_desc));
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = 512;
    image_desc.image_height = 512;
    cl_int err;

    cl_mem image_mem = clCreateImage(context.clContext().get(), flags, &image_format, &image_desc, nullptr, &err);
}