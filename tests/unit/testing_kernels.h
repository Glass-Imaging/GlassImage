const char* testing_kernel_code = R"(

typedef struct {
    int int_value;
    float float_value;
} CustomBufferStruct;

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void BufferAddKernel(__global float* data, float value) {
    int x = get_global_id(0);
    data[x] += value;
}

__kernel void CustomBufferAddKernel(__global CustomBufferStruct* data){
    int x = get_global_id(0);
    CustomBufferStruct s = data[x];
    s.float_value = s.float_value + s.int_value;
    data[x] = s;
}

__kernel void ImageAddKernel(read_only image2d_t image, float value, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);
    float4 pixel = read_imagef(image, sampler, (int2)(x, y));
    pixel = pixel + value;
    write_imagef(output, (int2)(x, y), pixel);
}

// Read values from a distance to test texture access speeds
__kernel void ReadIrregular2d(read_only image2d_t image, const int dist, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 pix = (float4)(0.0f);
    for(int yread = y - dist; yread <= y + dist; yread += dist){
        for(int xread = x - dist; xread <= x + dist; xread += dist){
            pix = pix + read_imagef(image, sampler, (int2)(xread, yread));
        }
    }
    write_imagef(output, (int2)(x, y), pix);
}

// Read values from a distance to test texture access speeds
__kernel void WriteIrregular2d(read_only image2d_t image, const int dist, write_only image2d_t output){
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 pix = (float4)(0.0f);
    for(int yread = y - dist; yread <= y + dist; yread += dist){
        for(int xread = x - dist; xread <= x + dist; xread += dist){
            pix = pix + read_imagef(image, sampler, (int2)(x, y));
        }
    }
    write_imagef(output, (int2)(x, y), pix);
}

)";
