const char* kernel_code = R"(
__kernel void add_one(__global float* data, int size) {
    int id = get_global_id(0);
    printf("Running id %i\n", id);
    if (id < size)
        data[id] += 1.0f;
}
)";
