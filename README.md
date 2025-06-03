# GlassImage
### A Modern C++ library for image processing with OpenCL -- for Android and Unix-like systems.
**Fabio Riccardi
Glass Imaging, Inc.
<fabio@glass-imaging.com>**

## Introduction

**GlassImage** is a C++ API to seamlessly use **OpenCL Compute** for GPGPU and imaging applications, designed to be used in conjunction with **[opencl.hpp](https://github.com/KhronosGroup/OpenCL-CLHPP)**, it provides:

* An Android bridge to link your application to the OpenCL libraries provided by the device's manufacturer.
* A Simple and powerful set of C++ classes to represent and manipulate typed images in memory, on the GPU, and on the file system.

## Motivation

OpenCL is a very powerful and mature GPU API, available on a variety of platforms, it allows for a very portable way to write high performance applications using GPU acceleration.

On Android devices OpenCL is still the best way to get the best performance out of the GPU. OpenCL is a mature solution with an excellent shading language and lots of advanced tools for control on numeric precision and performance.

Alternatives are OpenGL, which is quite old and clunky, and Vulkan, which could be great except that basic fundamental tools — such as the shader compiler — are still in their infancy.

One of the biggest hurdles to using OpenCL on Android devices is how to link your application to the libraries provided by the device manufacturer. To this day there seems to be no official way to do that.

GlassImage provides a transparent bridge to the OpenCL libraries installed by the device manufacturer, allowing the use of powerful header based wrappers such as opencl.hpp

OpenCL applications need to create and manage opaque objects (CLImage, CLBuffer) that represent GPU data, and connect this data to the GPU shaders that manipulate them. Even with the help of libraties like opencl.hpp, managing OpenCL memory objects tends to be very error prone and a source of hard to debug issues.

GlassImage provides high level typed C++ wrapper objects that allow to tie the GPU memory objects with their omologous in the CPU memory space.

## Requirements

Reasonably recent versions **Android Studio** and **Xcode** are needed to build and run the examples, on macOS you will need to install **libpng**, **libjpeg** and **zlib** using **[homebrew](https://brew.sh)** to read and write JPEG and PNG files:

    brew install libpng libjpeg zlib

For Android I have included prebuilt versions of the libpng, libjpeg-turbo, and zlib, with their respective header files.

## An Example

To understand this example some familiarity is expected with **[Modern C++](https://docs.microsoft.com/en-us/cpp/cpp/welcome-back-to-cpp-modern-cpp)**, **[OpenCL](https://github.com/KhronosGroup/OpenCL-Guide)** and **[opencl.hpp](https://github.com/KhronosGroup/OpenCL-CLHPP)**.

Two example projects are provided to demonstrate GlassImage's use with Android Studio and Xcode. The Android project generates both a command line tool and an Android App, the Xcode project generates a command line tool for macOS.

Let's look at the `main.cc` file of the command line tool:

```c++
    auto inputImage = gls::image<gls::rgba_pixel>::read_png_file(filePath);
```

reads a PNG image from the file system and returns a `std::uinique_ptr` to an in memory RGBA representation
of the image.

A variety of image formats are available for the most common image layouts (Luma, LumaAlpha, RGB, RGBA)
and data formats (uint8, uint16, float32 and float16), image formats and pixel data types are defined in
`gls_image.hpp`.

To create an image that can be directly accessed by OpenCL we can use:

```c++
    gls::cl_image_2d<gls::rgba_pixel> clInputImage(context, *inputImage);
```

Similarly to `gls::image`, `gls::cl_image_2d` will create an image-like object that acts as a wrapper to the OpenCL texture image, initialized with the geometry, data type and data content of `inputImage`. OpenCL image wrappers are available for CLImage2D, CLImage2DArray, CLImage3D and CLImage2D backed by CLBuffers, and they are defined in `gls_cl_image.hpp`.

All image objects are *strongly typed*, making it possible to use the C++ type system to robustly interface between C++
and **OpenCL/opencl.hpp**.

To get the underlying OpenCL Image2D object, the getImage2D() accessor is provided. The following code fragmnent
from `cl_pipeline.cpp` illustrates how to use the `gls::cl_image` types in conjunction with **[opencl.hpp](https://github.com/KhronosGroup/OpenCL-CLHPP)**:

```c++
    int blur(const gls::cl_image_2d<gls::rgba_pixel>& input, gls::cl_image_2d<gls::rgba_pixel>* output) {
        try {
            // Load the shader source
            const auto blurProgram = gls::loadOpenCLProgram("blur");

            // Bind the kernel parameters
            auto blurKernel = cl::KernelFunctor<cl::Image2D,  // input
                                                cl::Image2D   // output
                                                >(*blurProgram, "blur");

            // Schedule the kernel on the GPU
            blurKernel(gls::buildEnqueueArgs(output->width, output->height),
                       input.getImage2D(), output->getImage2D());
            return 0;
        } catch (cl::Error& err) {
            gls::logging::LogError(TAG) << "Caught Exception: " << std::string(err.what())
                           << " - " << gls::clStatusToString(err.err())
                           << std::endl;
            return -1;
        }
    }
```

To retrieve data from a cl_image, both memory mapping and memory copy are available, for instance:

```c++
    auto outputImage = clOutputImage.mapImage();
    outputImage.write_png_file("output.png");
    clOutputImage.unmapImage(outputImage);
```

Creates a *zero-copy* `gls::image` instance wrapped around the `gls::cl_image` OpenCL representation, the image is written to the file system as a PNG file and then unmapped from memory.

## More about `gls::image`

gls::image objects efficiently represent 2D images in memory. The following example shows how to generate a RGBA image from a RGB:

```c++
    void convertRGBtoRGBA(const gls::image<gls::rgb_pixel>& input, gls::image<gls::rgba_pixel>* output) {
        for (int y = 0; y < input.height; y++) {
            for (int x = 0; x < input.width; x++) {
                const gls::rgb_pixel& p = input[y][x];
                (*output)[y][x] = gls::rgba_pixel(p.red, p.green, p.blue, 255);
            }
        }
    }
```

We can see that gls::image objects have a **width** and a **height** and can be accessed using array subscript **[ ]** operators as 2D matrices. The first subscript (`input[y]`) returns a row of the image, the second allows to access individual pixels (`input[y][x]`). Notice that in this code fragment `p` is a reference to a pixel value within the `input` image data store.

Pixels are strongly typed and several common basic types are predifined (in `cl_image.hpp`), pixel components are accessed with **red**, **green**, **blue** and **alpha**, or as a vector (e.g: `p[0]`).

`gls::image` objects are optimized for performance and minimal memory footprint, utilizing modern C++ features for object allocation and lifetime. `gls::image` objects can be allocated directly on the stack or through factory methods returning a `std::unique_ptr` to the object. 

Image IO functionality is available for **PNG** and **JPEG** files. The following example shows how to save a single channel floating point image to a JPEG file:

```c++
    void saveNormalizedMap(const gls::image<gls::luma_pixel_float>& image,
                           const std::string& outfile) {
         // Find Minimum and Maximum values in the image
        float max = std::numeric_limits<float>::min();
        float min = std::numeric_limits<float>::max();
        for (const auto& val : image.pixels()) {
            if (val.luma < min) {
                min = val.luma;
            } else if (val.luma > max) {
                max = val.luma;
            }
        }
        float multiplier = (max - min) > 0 ? 255 / (max - min) : 1;

        // Create a single channel 8-bit image with values normalized in the range [0-255]
        gls::image<gls::luma_pixel> outputImage(image.width, image.height);
        for (int y = 0; y < outputImage.height; y++) {
            for (int x = 0; x < outputImage.width; x++) {
                outputImage[y][x].luma = (int) (multiplier * (image[y][x].luma - min));
            }
        }
        // Save it as a JPEG file
        outputImage.write_jpeg_file(outfile, 95);
    }
```

## More about `gls::cl_image_*` and `OpenCL` shaders

The `gls::cl_image_*` constructors automatically create `opencl.hpp` cl::Image objects with the right format and memory access flags using the type information provided by the pixel type.

It is possible to create and manipulate images in variety of layouts (Luma, LumaAlpha, RGB, RGBA) and data formats (uint8, uint16, float32 and float16), image formats and pixel data types are defined in `gls_image.hpp`.

Once bound to shaders OpenCL will access the image data as normalized (0-1) floating point data for `integer` image textures, and unnormalized float data for `floating point` image textures.

The following code fragment shows the simple OpenCL blur kernel from our example app:

```c++
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    float3 boxBlur(image2d_t blurMap, int2 imageCoordinates) {
        const int filterSize = 15;
        float3 blur = 0;
        for (int y = -filterSize / 2; y <= filterSize / 2; y++) {
            for (int x = -filterSize / 2; x <= filterSize / 2; x++) {
                int2 sampleCoordinate = imageCoordinates + (int2)(x, y);
                float3 blurSample = read_imagef(blurMap, sampler, sampleCoordinate).xyz;
                blur += blurSample;
            }
        }
        return blur / (filterSize * filterSize);
    }

    kernel void blur(read_only image2d_t input, write_only image2d_t output) {
        const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
        float3 result = boxBlur(input, imageCoordinates);
        write_imagef(output, imageCoordinates, (float4) (result, 1));
    }
```

Compare this code with the blur() function presented above which binds the gls::cl_image_2d objects to the OpenCL kernel.

## Putting It All Together

The full code from main.cpp shows how everything connects together:

```c++
    int main(int argc, const char* argv[]) {
        printf("Hello GlassImage!\n");

        if (argc > 1) {
            // Initialize the OpenCL environment and get the context
            cl::Context context = gls::getContext();

            // Read the input file into an image object
            auto inputImage = gls::image<gls::rgba_pixel>::read_png_file(argv[1]);

            // Load image data in OpenCL image texture
            gls::cl_image_2d<gls::rgba_pixel> clInputImage(context, *inputImage);

            // Output Image from OpenCL processing
            gls::cl_image_2d<gls::rgba_pixel> clOutputImage(context, clInputImage.width, clInputImage.height);

            // Execute OpenCL Blur algorithm
            blur(clInputImage, &clOutputImage);

            // Use OpenCL's memory mapping for zero-copy image Output
            auto outputImage = clOutputImage.mapImage();
            outputImage.write_png_file("output.png");
            clOutputImage.unmapImage(outputImage);
        }
    }
```

## Compilation
### MacOS
```
cmake -B build . --preset mac
cmake --build build --parallel 8 --target GlassImage
```

### Cross-compile to Android
Make sure the toolchain file is correct in your preset.
```
cmake -B build . --preset android
cmake --build build --parallel 8 --target GlassImage
```
