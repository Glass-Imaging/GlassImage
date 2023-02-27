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

/*
 Derived from https://www.johndcook.com/blog/skewness_kurtosis/
 and https://www.osti.gov/servlets/purl/1028931
 TODO: Revise this to: https://www.osti.gov/servlets/purl/1426900
 */

#ifndef gls_statistics_h
#define gls_statistics_h

#include <cmath>

namespace gls {

template <typename T>
class statistics {
   public:
    statistics() { clear(); }

    void clear() {
        n = 0;
        M1 = M2 = M3 = M4 = 0.0;
    }

    void push(T x) {
        T delta, delta_n, delta_n2, term1;

        long long n1 = n;
        n++;
        delta = x - M1;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        M1 += delta_n;
        M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
        M2 += term1;
    }

    long long numDataValues() const { return n; }

    T mean() const { return M1; }

    T variance() const { return M2 / (n - 1.0); }

    T standardDeviation() const { return std::sqrt(variance()); }

    T skewness() const { return std::sqrt(double(n)) * M3 / pow(M2, 1.5); }

    T kurtosis() const { return double(n) * M4 / (M2 * M2) - 3.0; }

    statistics& operator+=(const statistics& rhs) {
        statistics combined = *this + rhs;
        *this = combined;
        return *this;
    }

    friend statistics operator+(const statistics a, const statistics b);

   private:
    long long n;
    T M1, M2, M3, M4;
};

template <typename T>
statistics<T> operator+(const statistics<T> a, const statistics<T> b) {
    statistics<T> combined;

    combined.n = a.n + b.n;

    T delta = b.M1 - a.M1;
    T delta2 = delta * delta;
    T delta3 = delta * delta2;
    T delta4 = delta2 * delta2;

    combined.M1 = (a.n * a.M1 + b.n * b.M1) / combined.n;

    combined.M2 = a.M2 + b.M2 + delta2 * a.n * b.n / combined.n;

    combined.M3 = a.M3 + b.M3 + delta3 * a.n * b.n * (a.n - b.n) / (combined.n * combined.n);
    combined.M3 += 3.0 * delta * (a.n * b.M2 - b.n * a.M2) / combined.n;

    combined.M4 =
        a.M4 + b.M4 + delta4 * a.n * b.n * (a.n * a.n - a.n * b.n + b.n * b.n) / (combined.n * combined.n * combined.n);
    combined.M4 += 6.0 * delta2 * (a.n * a.n * b.M2 + b.n * b.n * a.M2) / (combined.n * combined.n) +
                   4.0 * delta * (a.n * b.M3 - b.n * a.M3) / combined.n;

    return combined;
}

}  // namespace gls

#endif /* gls_statistics_h */
