#include <vector>
#include <cstddef>

void compute_bound_no_simd(const std::vector<double>& x,
                           std::vector<double>& y)
{
    const std::size_t N = x.size();
#pragma clang loop vectorize(disable)
    
    for (size_t i = 0; i < N; i++) {
        y[i] = x[i];
    }

    for (size_t t = 0; t < 50; t++) {
        for (size_t i = 0; i < N; i++) {
            y[i] = y[i] * y[i];
        }
    }
}
