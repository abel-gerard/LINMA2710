#include "matrix.hpp"
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>


Matrix::Matrix(int rows, int cols)
    : rows(0), cols(0)
{
    this->rows = rows;
    this->cols = cols;
    this->data = std::vector<double>(rows * cols);
}

Matrix::Matrix(const Matrix &other)
    : rows(0), cols(0)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->data = other.data;
}

int Matrix::numRows() const
{
    return rows;
}

int Matrix::numCols() const
{
    return cols;
}

double Matrix::get(int i, int j) const
{
    return data[i*cols+j];
}

void Matrix::set(int i, int j, double value)
{
    data[i*cols+j] = value;
}

void Matrix::fill(double value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i*cols+j] = value;
        }
    }
}

Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix nu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nu.data[i*cols+j] = data[i*cols+j] + other.data[i*cols+j];
        }
    } 

    return nu;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix nu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nu.data[i*cols+j] = data[i*cols+j] - other.data[i*cols+j];
        }
    } 

    return nu;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols != other.rows) {
        throw std::invalid_argument("Incompatible dimensions for multiplication");
    }

    Matrix nu(rows, other.cols);
    const Matrix t_other = other.transpose();
    
#ifdef USE_OMP
    #pragma omp parallel for collapse(2) schedule(static) num_threads(OMP_T)
#endif
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {

            double acc = 0.0;
            const double *data_self = &data[i*cols];
            const double *data_other = &t_other.data[j*t_other.cols];
         
#ifdef USE_AVX2
            __m256d vec_acc1 = _mm256_setzero_pd();
            __m256d vec_acc2 = _mm256_setzero_pd();
            __m256d vec_acc3 = _mm256_setzero_pd();
            __m256d vec_acc4 = _mm256_setzero_pd();

            int k = 0;
            for (; k <= cols - 16; k += 16) { // Pipelining 4 AVX2 FMA operations
                vec_acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(data_self+k+0), _mm256_loadu_pd(data_other+k+0), vec_acc1);
                vec_acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(data_self+k+4), _mm256_loadu_pd(data_other+k+4), vec_acc2);
                vec_acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(data_self+k+8), _mm256_loadu_pd(data_other+k+8), vec_acc3);
                vec_acc4 = _mm256_fmadd_pd(_mm256_loadu_pd(data_self+k+12), _mm256_loadu_pd(data_other+k+12), vec_acc4);
            }

            // Horizontal add of the 4 accumulators
            __m256d vec_acc = _mm256_add_pd(
                _mm256_add_pd(vec_acc1, vec_acc2),
                _mm256_add_pd(vec_acc3, vec_acc4)
            );

            // Reduction
            __m128d lo = _mm256_castpd256_pd128(vec_acc);
            __m128d hi = _mm256_extractf128_pd(vec_acc, 1);
            __m128d sum = _mm_add_pd(lo, hi);

            __m128d shuf = _mm_unpackhi_pd(sum, sum);
            acc = _mm_cvtsd_f64(_mm_add_sd(sum, shuf));

            for (; k < cols; k++) {
                acc += data_self[k] * data_other[k];
            }

            nu.data[i*nu.cols+j] = acc;
#else
#ifdef USE_OMP
                #pragma omp parallel for simd reduction(+:acc) num_threads(OMP_T)
#else
                #pragma GCC ivdep
                #pragma GCC unroll 4
#endif
            for (int k = 0; k < cols; k++) {
                acc += data_self[k] * data_other[k];
            }
            nu.data[i*nu.cols+j] = acc;
#endif
        }
    }  

    return nu;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix nu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nu.data[i*cols+j] = data[i*cols+j] * scalar;
        }
    } 

    return nu;
}

Matrix Matrix::transpose() const
{
    Matrix nu(cols, rows);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nu.data[j*nu.cols+i] = data[i*cols+j];
        }
    } 

    return nu;
}

Matrix Matrix::apply(const std::function<double(double)> &func) const
{
    Matrix nu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nu.data[i*cols+j] = func(data[i*cols+j]);
        }
    } 

    return nu;
}

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i*cols+j] -= other.data[i*cols+j] * scalar;
        }
    } 
}
