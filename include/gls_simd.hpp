//
//  gls_simd.hpp
//  GlassCamera
//
//  Created by Fabio Riccardi on 8/23/23.
//

#ifndef gls_simd_h
#define gls_simd_h

#include "gls_linalg.hpp"
#include "float16.hpp"

namespace gls {

template <size_t N, typename T = float, size_t NV = N == 3 ? 4 : N>
requires(N == 2 || N == 4 || N == 8 || N == 16)
struct simdVector : public std::array<T, NV> {

    simdVector() : std::array<T, NV>() { }

    template <size_t N2, typename T2>
    simdVector(const std::array<T2, N2>& other) {
        for (int i = 0; i < std::min(N, N2); i++) {
            (*this)[i] = other[i];
        }
    }
} __attribute__ ((aligned(NV * sizeof(T))));

typedef simdVector<2, int> int2;
typedef simdVector<4, int> int4;
typedef int4 int3;
typedef simdVector<8, int> int8;
typedef simdVector<16, int> int16;

typedef simdVector<2, uint> uint2;
typedef simdVector<4, uint> uint4;
typedef uint4 uint3;
typedef simdVector<8, uint> uint8;
typedef simdVector<16, uint> uint16;

typedef simdVector<2, float> float2;
typedef simdVector<4, float> float4;
typedef float4 float3;
typedef simdVector<8, float> float8;
typedef simdVector<16, float> float16;

typedef simdVector<2, half> half2;
typedef simdVector<4, half> half4;
typedef half4 half3;
typedef simdVector<8, half> half8;
typedef simdVector<16, half> half16;

template <size_t N, typename T = float>
requires(N == 2 || N == 3 || N == 4 || N == 8 || N == 16)
struct simdMatrix {
    std::array<simdVector<N == 3 ? 4 : N, T>, N> m;

    simdMatrix(const gls::Matrix<N, N>& transform) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                m[j][i] = transform[j][i];
            }
        }
    }
};

typedef simdMatrix<3, float> float3x3;
typedef simdMatrix<4, float> float4x4;

typedef simdMatrix<3, half> half3x3;
typedef simdMatrix<4, half> half4x4;

}  // namespace gls

#endif /* gls_simd_h */
