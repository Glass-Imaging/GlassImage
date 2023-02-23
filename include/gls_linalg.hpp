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

#ifndef gls_linalg_h
#define gls_linalg_h

#include <array>
#include <span>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cassert>

#define DEBUG_ARRAY_INITIALIZATION false

namespace gls {

template<size_t M, size_t N, typename value_type = float> struct Matrix;

// ---- Vector Type ----
template <size_t N, typename value_type = float>
struct Vector : public std::array<value_type, N> {
#if DEBUG_ARRAY_INITIALIZATION
    Vector() {
        this->fill(0);
    }
#else
    Vector() = default;
#endif

    Vector(value_type val) {
        this->fill(val);
    }

    Vector(const std::array<value_type, N>& v) {
        assert(v.size() == N);
        std::copy(v.begin(), v.end(), this->begin());
    }

    Vector(const value_type(&il)[N]) {
        std::copy(il, il + N, this->begin());
    }

    Vector(const std::vector<value_type>& v) {
        assert(v.size() == N);
        std::copy(v.begin(), v.end(), this->begin());
    }

    Vector(std::initializer_list<value_type> list) {
        assert(list.size() == N);
        std::copy(list.begin(), list.end(), this->begin());
    }

    template <typename T>
    Vector(const std::array<T, N>& v) {
        assert(v.size() == N);
        for (int i = 0; i < N; i++) {
            (*this)[i] = v[i];
        }
    }

    template<size_t P, size_t Q>
    requires (P * Q == N)
    Vector(const Matrix<P, Q>& m) {
        const auto ms = m.span();
        std::copy(ms.begin(), ms.end(), this->begin());
    }

    static Vector<N, value_type> zeros() {
        Vector<N, value_type> v;
        v.fill(0);
        return v;
    }

    static Vector<N, value_type> ones() {
        Vector<N, value_type> v;
        v.fill(1);
        return v;
    }

    template <typename T>
    Vector& operator += (const T& v) {
        *this = *this + v;
        return *this;
    }

    template <typename T>
    Vector& operator -= (const T& v) {
        *this = *this - v;
        return *this;
    }

    template <typename T>
    Vector& operator *= (const T& v) {
        *this = *this * v;
        return *this;
    }

    template <typename T>
    Vector& operator /= (const T& v) {
        *this = *this / v;
        return *this;
    }

    // Cast to a const value_type*
    operator const value_type*() const {
        return this->data();
    }

    // Cast to a Vector with a different value_type
    template <typename T>
    operator gls::Vector<N, T>() const {
        gls::Vector<N, T> result;
        for (int j = 0; j < N; j++) {
            result[j] = (T) (*this)[j];
        }
        return result;
    }

    // Cast Vector elements to a Matrix of compatible dimensions and of any given value_type T
    template <size_t K, size_t M, typename T>
    requires (K * M == N)
    operator gls::Matrix<K, M, T>() const {
        gls::Matrix<K, M, T> result;
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < M; i++) {
                result[j][i] = (T) (*this)[j * M + i];
            }
        }
        return result;
    }
};

template <size_t N> using DVector = Vector<N, double>;

// Vector - Vector Addition (component-wise)
template <size_t N, typename value_type>
inline Vector<N, value_type> operator + (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    auto ita = a.begin();
    auto itb = b.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&](value_type &r){ r = *ita++ + *itb++; });
    return result;
}

// Vector - Vector Subtraction (component-wise)
template <size_t N, typename value_type>
inline Vector<N, value_type> operator - (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    auto ita = a.begin();
    auto itb = b.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&](value_type &r){ r = *ita++ - *itb++; });
    return result;
}

// Vector - Vector Multiplication (component-wise)
template <size_t N, typename value_type>
inline Vector<N, value_type> operator * (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    auto ita = a.begin();
    auto itb = b.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&](value_type &r){ r = *ita++ * *itb++; });
    return result;
}

// Vector Dot Product
template <size_t N, typename value_type>
inline value_type dot(const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    value_type result = 0;
    for (size_t i = 0; i < N; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Vector - Vector Division (component-wise)
template <size_t N, typename value_type>
inline Vector<N, value_type> operator / (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    auto ita = a.begin();
    auto itb = b.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&](value_type &r){ r = *ita++ / *itb++; });
    return result;
}

// Vector - Scalar Addition
template <size_t N, typename value_type>
inline Vector<N, value_type> operator + (const Vector<N, value_type>& v, value_type a) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = *itv++ + a; });
    return result;
}

// Vector - Scalar Addition (commutative)
template <size_t N, typename value_type>
inline Vector<N, value_type> operator + (value_type a, const Vector<N, value_type>& v) {
    return v + a;
}

// Vector - Scalar Subtraction
template <size_t N, typename value_type>
inline Vector<N, value_type> operator - (const Vector<N, value_type>& v, value_type a) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = *itv++ - a; });
    return result;
}

// Scalar - Vector Subtraction
template <size_t N, typename value_type>
inline Vector<N, value_type> operator - (value_type a, const Vector<N, value_type>& v) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = a - *itv++; });
    return result;
}

// Vector - Scalar Multiplication
template <size_t N, typename value_type>
inline Vector<N, value_type> operator * (const Vector<N, value_type>& v, value_type a) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = *itv++ * a; });
    return result;
}

// Scalar - Vector Multiplication (commutative)
template <size_t N, typename value_type>
inline Vector<N, value_type> operator * (value_type a, const Vector<N, value_type>& v) {
    return v * a;
}

// Vector - Scalar Division
template <size_t N, typename value_type>
inline Vector<N, value_type> operator / (const Vector<N, value_type>& v, value_type a) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = *itv++ / a; });
    return result;
}

// Scalar - Vector Division
template <size_t N, typename value_type>
inline Vector<N, value_type> operator / (value_type a, const Vector<N, value_type>& v) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = a / *itv++; });
    return result;
}

template <size_t N, typename value_type>
inline Vector<N, value_type> abs(const Vector<N, value_type>& v) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&itv](value_type &r){ r = std::abs(*itv++); });
    return result;
}

// Vector - Scalar Max
template <size_t N, typename value_type>
inline Vector<N, value_type> max(const Vector<N, value_type>& v, value_type a) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = std::max(*itv++, a); });
    return result;
}

// Vector - Scalar Min
template <size_t N, typename value_type>
inline Vector<N, value_type> min(const Vector<N, value_type>& v, value_type a) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&a, &itv](value_type &r){ r = std::min(*itv++, a); });
    return result;
}

// Vector - Square Root
template <size_t N, typename value_type>
inline Vector<N, value_type> sqrt(const Vector<N, value_type>& v) {
    auto itv = v.begin();
    Vector<N, value_type> result;
    std::for_each(result.begin(), result.end(), [&itv](value_type &r){ r = std::sqrt(*itv++); });
    return result;
}

template <size_t N>
inline Vector<N, bool> operator ! (const Vector<N, bool>& a) {
    auto ita = a.begin();
    Vector<N, bool> result;
    std::for_each(result.begin(), result.end(), [&](bool &r){ r = !*ita++; });
    return result;
}

// Vector - Vector comparison (component-wise)
template <size_t N, typename value_type>
inline Vector<N, bool> operator < (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    auto ita = a.begin();
    auto itb = b.begin();
    Vector<N, bool> result;
    std::for_each(result.begin(), result.end(), [&](bool &r){ r = *ita++ < *itb++; });
    return result;
}

template <size_t N, typename value_type>
inline Vector<N, bool> operator > (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    return b < a;
}

template <size_t N, typename value_type>
inline Vector<N, bool> operator <= (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    return !(a > b);
}

template <size_t N, typename value_type>
inline Vector<N, bool> operator >= (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    return !(a < b);
}

template <size_t N, typename value_type>
inline Vector<N, bool> operator == (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    auto ita = a.begin();
    auto itb = b.begin();
    Vector<N, bool> result;
    std::for_each(result.begin(), result.end(), [&](bool &r){ r = *ita++ == *itb++; });
    return result;
}

template <size_t N, typename value_type>
inline Vector<N, bool> operator != (const Vector<N, value_type>& a, const Vector<N, value_type>& b) {
    return !(a == b);
}

template <size_t N, typename value_type>
inline Vector<N, bool> isnan(const Vector<N, value_type>& a) {
    auto ita = a.begin();
    Vector<N, bool> result;
    std::for_each(result.begin(), result.end(), [&](bool &r){ r = std::isnan(*ita++); });
    return result;
}

template <size_t N>
inline bool all(const Vector<N, bool>& a) {
    bool result = true;
    std::for_each(a.begin(), a.end(), [&result](const bool &v){ result = result && v; });
    return result;
}

template <size_t N>
inline bool any(const Vector<N, bool>& a) {
    bool result = false;
    std::for_each(a.begin(), a.end(), [&result](const bool &v){ result = result || v; });
    return result;
}

// ---- Matrix Type ----

template <size_t N, size_t M, typename value_type>
struct Matrix : public std::array<Vector<M, value_type>, N> {
    // Give Matrix some Image traits
    static const constexpr int width = M;
    static const constexpr int height = N;

    Matrix() { }

    Matrix(const Vector<N * M, value_type>& v) {
        std::copy(v.begin(), v.end(), span().begin());
    }

    Matrix(const value_type(&il)[N * M]) {
        std::copy(il, il + (N * M), span().begin());
    }

    Matrix(const std::array<value_type, M>(&il)[N]) {
        // This is safe, il is just an array of arrays
        std::copy((value_type *) il, (value_type *) il + (N * M), span().begin());
    }

    Matrix(const std::vector<value_type>& v) {
        assert(v.size() == N * M);
        std::copy(v.begin(), v.end(), span().begin());
    }

    Matrix(std::initializer_list<value_type> list) {
        assert(list.size() == N * M);
        std::copy(list.begin(), list.end(), span().begin());
    }

    Matrix(std::initializer_list<std::array<value_type, M>> list) {
        assert(list.size() == N);
        size_t row = 0;
        for (const auto& v : list) {
            std::copy(v.begin(), v.end(), span(row++).begin());
        }
    }

    void fill(value_type val) {
        for (int i = 0; i < N; i++) {
            (*this)[i].fill(val);
        }
    }

    static Matrix<N, M, value_type> zeros() {
        Matrix<N, M, value_type> m;
        m.fill(0);
        return m;
    }

    static Matrix<N, M, value_type> ones() {
        Matrix<N, M, value_type> m;
        m.fill(0);
        return m;
    }

    static Matrix<N, M, value_type> identity() {
        Matrix<N, M, value_type> m;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                m[i][j] = i == j ? 1 : 0;
            }
        }
        return m;
    }

    template <typename T>
    Matrix& operator += (const T& v) {
        for (int i = 0; i < N; i++) {
            (*this)[i] += v;
        }
        return *this;
    }

    template <typename T>
    Matrix& operator -= (const T& v) {
        for (int i = 0; i < N; i++) {
            (*this)[i] -= v;
        }
        return *this;
    }

    template <typename T>
    Matrix& operator *= (const T& v) {
        for (int i = 0; i < N; i++) {
            (*this)[i] *= v;
        }
        return *this;
    }

    template <typename T>
    Matrix& operator /= (const T& v) {
        for (int i = 0; i < N; i++) {
            (*this)[i] /= v;
        }
        return *this;
    }

    // Matrix Raw Data
    std::span<value_type> span() {
        return std::span(&(*this)[0][0], N * M);
    }

    const std::span<const value_type> span() const {
        return std::span(&(*this)[0][0], N * M);
    }

    // Matrix Row Raw Data
    std::span<value_type> span(size_t row) {
        return std::span(&(*this)[row][0], M);
    }

    const std::span<const value_type> span(size_t row) const {
        return std::span(&(*this)[row][0], M);
    }

    // Cast to a const value_type*
    operator const value_type*() const {
        return span().data();
    }

    typedef value_type (*opPtr)(value_type a, value_type b);

    // Cast to a Matrix with a different value_type
    template <typename T>
    operator gls::Matrix<N, M, T>() const {
        gls::Matrix<N, M, T> result;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                result[j][i] = (T) (*this)[j][i];
            }
        }
        return result;
    }

    // Cast Matrix elements to a Vector of any given value_type T
    template <typename T>
    operator gls::Vector<N * M, T>() const {
        gls::Vector<N * M, T> result;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                result[j * M + i] = (T) (*this)[j][i];
            }
        }
        return result;
    }
};

template <size_t N, size_t M> using DMatrix = Matrix<N, M, double>;

template <size_t N, size_t M, typename value_type>
std::span<value_type> span(Matrix<N, M, value_type>& m) {
    return std::span(&m[0][0], N * M);
}

template <size_t N, size_t M, typename value_type>
const std::span<const value_type> span(const Matrix<N, M, value_type>& m) {
    return std::span(&m[0][0], N * M);
}

// Matrix Transpose
template<size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> transpose(const Matrix<M, N, value_type>& m) {
    Matrix<N, M, value_type> result;
    for (size_t j = 0; j < M; j++) {
        for (size_t i = 0; i < N; i++) {
            result[i][j] = m[j][i];
        }
    }
    return result;
}

// General Matrix Multiplication
template <size_t N, size_t K, size_t M, typename value_type>
inline Matrix<M, N, value_type> operator * (const Matrix<M, K, value_type>& a, const Matrix<K, N, value_type>& b) {
    Matrix<M, N, value_type> result;
    const auto bt = transpose(b);
    for (size_t j = 0; j < M; j++) {
        for (size_t i = 0; i < N; i++) {
            value_type sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += a[j][k] * bt[i][k];
            }
            result[j][i] = sum;
        }
    }
    return result;
}

// Matrix - Vector Multiplication
template <size_t M, size_t N, typename value_type>
inline Vector<M, value_type> operator * (const Matrix<M, N, value_type>& a, const Vector<N, value_type>& b) {
    const auto result = a * Matrix<N, 1, value_type> { b };
    return Vector<M, value_type>(result);
}

// Vector - Matrix Multiplication
template <size_t M, size_t N, typename value_type>
inline Vector<N, value_type> operator * (const Vector<M, value_type>& a, const Matrix<M, N, value_type>& b) {
    const auto result = Matrix<1, N, value_type> { a } * b;
    return Vector<N, value_type>(result);
}

// (Square) Matrix Division (Multiplication with Inverse)
template <size_t N, typename value_type>
inline Matrix<N, N, value_type> operator / (const Matrix<N, N, value_type>& a, const Matrix<N, N, value_type>& b) {
    return a * inverse(b);
}

// Iterate over the elements of the input and output matrices applying a Matrix-Matrix function
template<size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> apply(const Matrix<M, N, value_type>& a, const Matrix<M, N, value_type>& b, typename Matrix<N, M, value_type>::opPtr f) {
    Matrix<N, M, value_type> result;
    auto ita = span(a).begin();
    auto itb = span(b).begin();
    for (auto& r : span(result)) {
        r = f(*ita++, *itb++);
    }
    return result;
}

// Iterate over the elements of the input and output matrices applying a Matrix-Scalar function
template<size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> apply(const Matrix<M, N, value_type>& a, value_type b, typename Matrix<N, M, value_type>::opPtr f) {
    Matrix<N, M, value_type> result;
    auto ita = span(a).begin();
    for (auto& r : span(result)) {
        r = f(*ita++, b);
    }
    return result;
}

// Matrix-Scalar Multiplication
template <size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> operator * (const Matrix<N, M, value_type>& a, value_type b) {
    return apply(a, b, [](value_type a, value_type b) {
        return a * b;
    });
}

// Matrix-Scalar Division
template <size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> operator / (const Matrix<N, M, value_type>& a, value_type b) {
    return apply(a, b, [](value_type a, value_type b) {
        return a / b;
    });
}

// Matrix-Matrix Addition
template <size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> operator + (const Matrix<N, M, value_type>& a, const Matrix<N, M, value_type>& b) {
    return apply(a, b, [](value_type a, value_type b) {
        return a + b;
    });
}

// Matrix-Scalar Addition
template <size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> operator + (const Matrix<N, M, value_type>& a, value_type b) {
    return apply(a, b, [](value_type a, value_type b) {
        return a + b;
    });
}

// Matrix-Matrix Subtraction
template <size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> operator - (const Matrix<N, M, value_type>& a, const Matrix<N, M, value_type>& b) {
    return apply(a, b, [](value_type a, value_type b) {
        return a - b;
    });
}

// Matrix-Scalar Subtraction
template <size_t N, size_t M, typename value_type>
inline Matrix<N, M, value_type> operator - (const Matrix<N, M, value_type>& a, value_type b) {
    return apply(a, b, [](value_type a, value_type b) {
        return a - b;
    });
}

// --- Matrix Inverse Support ---

// Cofactor Matrix
// https://en.wikipedia.org/wiki/Minor_(linear_algebra)#Inverse_of_a_matrix
template <size_t N1, size_t N2 = N1 - 1, typename value_type>
inline Matrix<N2, N2> cofactor(const Matrix<N1, N1, value_type>& m, size_t p, size_t q) {
    assert(p < N1 && q < N1);

    Matrix<N2, N2> result;

    // Looping for each element of the matrix
    size_t i = 0, j = 0;
    for (size_t row = 0; row < N1; row++) {
        for (size_t col = 0; col < N1; col++) {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != p && col != q) {
                result[i][j++] = m[row][col];

                // Row is filled, so increase row index and
                // reset col index
                if (j == N1 - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
    return result;
}

// Matrix Determinant using Laplace's Cofactor Expansion
// https://en.wikipedia.org/wiki/Minor_(linear_algebra)#Cofactor_expansion_of_the_determinant
template <size_t N, typename value_type>
inline value_type determinant(const Matrix<N, N, value_type>& m) {
    assert(N > 1);

    value_type sign = 1;
    value_type result = 0;
    // Iterate for each element of first row
    for (size_t f = 0; f < N; f++) {
        result += sign * m[0][f] * determinant(cofactor(m, 0, f));
        // terms are to be added with alternate sign
        sign = -sign;
    }
    return result;
}

// Matrix Determinant, Special case for size 1x1
template <typename value_type>
inline value_type determinant(const Matrix<1, 1, value_type>& m) {
    return m[0][0];
}

// Matrix Adjoint (Tanspose of the Cofactor Matrix)
// https://en.wikipedia.org/wiki/Adjugate_matrix
template <size_t N, typename value_type>
inline Matrix<N, N, value_type> adjoint(const Matrix<N, N, value_type>& m) {
    assert(N > 1);

    Matrix<N, N, value_type> adj;

    value_type sign = 1;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            value_type d = determinant(cofactor(m, i, j));
            adj[j][i] = d != 0 ? sign * d : 0;
        }
    }
    return adj;
}

// Matrix Adjoint - Special case for size 1x1
template <typename value_type>
inline Matrix<1, 1, value_type> adjoint(const Matrix<1, 1, value_type>& m) {
    return { 1 };
}

// Inverse Matrix using Cramer's rule - Pretty much always slower than Gauss-Jordan
// https://en.wikipedia.org/wiki/Minor_(linear_algebra)#Inverse_of_a_matrix
template <size_t N, typename value_type>
inline Matrix<N, N, value_type> cramerInverse(const Matrix<N, N, value_type>& m) {
    value_type det = determinant(m);
    if (det == 0) {
        throw std::domain_error("Singular Matrix.");
    }

    Matrix<N, N, value_type> inverse;
    const auto adj = adjoint(m);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            inverse[i][j] = adj[i][j] / det;
        }
    }

    return inverse;
}

// --- Large Matrix Inversion using Gauss-Jordan ---

template <size_t N, size_t M, typename value_type>
void swap_rows(Matrix<N, M, value_type>& m, size_t i, size_t j) {
    for (size_t column = 0; column < M; column++)
        std::swap(m[i][column], m[j][column]);
}

// Convert matrix to reduced row echelon form
// NOTE: This is only numerically stable in double precision
template <size_t N, size_t M, typename value_type>
void to_reduced_row_echelon_form(Matrix<N, M, value_type>& m) {
    for (size_t row = 0, lead = 0; row < N && lead < M; ++row, ++lead) {
        auto i = row;
        while (m[i][lead] == 0) {
            if (++i == N) {
                i = row;
                if (++lead == M) {
                    return;
                }
            }
        }
        swap_rows(m, i, row);

        const auto f = m[row][lead];
        if (f == 0) {
            throw std::domain_error("Singular Matrix.");
        }
        // Divide row by f
        m[row] /= f;

        for (size_t j = 0; j < N; ++j) {
            if (j != row) {
                // Subtract current row multiplied by m[j][lead] from row j
                m[j] -= m[j][lead] * m[row];
            }
        }
    }
}

// Gauss-Jordan Matrix Inversion
template <size_t N, typename value_type>
inline Matrix<N, N, value_type> inverse(const Matrix<N, N, value_type>& m) {
    // NOTE: With float data this method is prone to gross errors
    Matrix<N, 2 * N, double> tmp;
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < N; ++column) {
            tmp[row][column] = m[row][column];
            tmp[row][column + N] = row == column ? 1 : 0;
        }
    }
    to_reduced_row_echelon_form(tmp);
    Matrix<N, N, value_type> inv;
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < N; ++column)
            inv[row][column] = tmp[row][column + N];
    }
    return inv;
}

// LU Decomposition Solver
// Based on Wikipidia's C code example: https://en.wikipedia.org/wiki/LU_decomposition#C_code_example
template <size_t N, typename value_type>
void LUPDecompose(Matrix<N, N, value_type>& A,
                  Vector<N + 1, int>& P) {
    for (int i = 0; i <= N; i++) {
        P[i] = i; // Unit permutation matrix, P[N] initialized with N
    }

    for (int i = 0; i < N; i++) {
        value_type maxA = 0.0;
        int imax = i;

        for (int k = i; k < N; k++) {
            value_type absA = std::abs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA <= std::numeric_limits<value_type>::epsilon()) {
            throw std::domain_error("Singular Matrix.");
        }

        if (imax != i) {
            // pivoting P
            std::swap(P[i], P[imax]);

            // pivoting rows of A
            swap_rows(A, i, imax);

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (int j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];
            for (int k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }
}

template <size_t N, typename A_type, typename value_type>
gls::Vector<N, value_type> LUPSolve(const Matrix<N, N, A_type>&A,
                                    const Vector<N + 1, int>& P,
                                    const Vector<N, value_type>& b) {
    Vector<N, value_type> x;

    for (int i = 0; i < N; i++) {
        x[i] = b[P[i]];
        for (int k = 0; k < i; k++) {
            x[i] -= A[i][k] * x[k];
        }
    }
    for (int i = N - 1; i >= 0; i--) {
        for (int k = i + 1; k < N; k++) {
            x[i] -= A[i][k] * x[k];
        }
        x[i] /= A[i][i];
    }
    return x;
}

template <size_t N, typename value_type>
Matrix<N, N, value_type> LUPInvert(const Matrix<N, N, value_type>& A, const Vector<N + 1, int>& P) {
    const Matrix<N, N, value_type> IA;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
            IA[i][j] /= A[i][i];
        }
    }
    return IA;
}

template <size_t N, typename value_type>
value_type LUPDeterminant(const Matrix<N, N, value_type>& A, const Vector<N + 1, int>& P) {
    value_type det = A[0][0];

    for (int i = 1; i < N; i++) {
        det *= A[i][i];
    }
    return (P[N] - N) % 2 == 0 ? det : -det;
}

template <size_t N, typename value_type>
gls::Vector<N, value_type> LUSolve(const gls::Matrix<N, N, value_type>& m,
                                   const gls::Vector<N, value_type>& b) {
    Matrix<N, N, double> A(m);
    Vector<N + 1, int> P;
    LUPDecompose(A, P);
    return LUPSolve(A, P, b);
}

// Alternative LU Decomposition Solver
// Based on Wikipidia's C# code example: https://en.wikipedia.org/wiki/LU_decomposition#C#_code_example
template <size_t N, typename value_type>
gls::Vector<N, value_type> LUSolveAlt(const gls::Matrix<N, N, value_type>& m,
                                      const gls::Vector<N, value_type>& b) {
    // decomposition of matrix
    gls::Matrix<N, N, double> lu;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum += lu[i][k] * lu[k][j];
            }
            lu[i][j] = m[i][j] - sum;
        }
        for (int j = i + 1; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum += lu[j][k] * lu[k][i];
            }
            lu[j][i] = (1 / lu[i][i]) * (m[j][i] - sum);
        }
    }

    // lu = L+U-I
    // find solution of Ly = b
    gls::Vector<N, double> y;
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int k = 0; k < i; k++) {
            sum += lu[i][k] * y[k];
        }
        y[i] = b[i] - sum;
    }
    // find solution of Ux = y
    gls::Vector<N, value_type> x;
    for (int i = N - 1; i >= 0; i--) {
        double sum = 0;
        for (int k = i + 1; k < N; k++) {
            sum += lu[i][k] * x[k];
        }
        x[i] = (1 / lu[i][i]) * (y[i] - sum);
    }
    return x;
}

// From DCRaw (https://www.dechifro.org/dcraw/)
template <size_t size, typename value_type>
Matrix<size, 3> pseudoinverse(const Matrix<size, 3, value_type>& in) {
    Matrix<3,6> work;

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 6; j++) {
            work[i][j] = j == i + 3;
        }
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < size; k++) work[i][j] += in[k][i] * in[k][j];
        }
    }
    for (size_t i = 0; i < 3; i++) {
        value_type num = work[i][i];
        for (size_t j = 0; j < 6; j++) work[i][j] /= num;
        for (size_t k = 0; k < 3; k++) {
            if (k == i) continue;
            num = work[k][i];
            for (size_t j = 0; j < 6; j++) work[k][j] -= work[i][j] * num;
        }
    }
    Matrix<size, 3> out;
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < 3; j++) {
            out[i][j] = 0;
            for (size_t k = 0; k < 3; k++) {
                out[i][j] += work[j][k + 3] * in[i][k];
            }
        }
    }
    return out;
}

// --- Utility Functions ---

template <size_t N, typename value_type>
inline std::ostream& operator<<(std::ostream& os, const Vector<N, value_type>& v) {
    for (size_t i = 0; i < N; i++) {
        os << v[i];
        if (i < N - 1) {
            os << ", ";
        }
    }
    return os;
}

template <size_t N, size_t M, typename value_type>
inline std::ostream& operator<<(std::ostream& os, const Matrix<N, M, value_type>& m) {
    for (size_t j = 0; j < N; j++) {
        os << m[j] << ",";
        if (j < N-1) {
            os << std::endl;
        }
    }
    return os;
}

}  // namespace gls

namespace std {

// Useful for printing a Matrix on a single line

template <typename value_type>
inline std::ostream& operator<<(std::ostream& os, const std::span<value_type>& s) {
    for (size_t i = 0; i < s.size(); i++) {
        os << s[i];
        if (i < s.size() - 1) {
            os << ", ";
        }
    }
    return os;
}

} // namespace gls

#endif /* gls_linalg_h */
