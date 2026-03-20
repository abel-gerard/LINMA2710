#include "matrix.hpp"
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif


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

#define MUL_METHOD 1
Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols != other.rows) {
        throw std::invalid_argument("Incompatible dimensions for multiplication");
    }

    Matrix nu(rows, other.cols);

    // ijk loop ~1µs
#if MUL_METHOD == 0
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {

            double acc = 0.;
            const double *const data_self = &data[i*cols];
            
            #pragma omp simd reduction(+:acc)
            for (int k = 0; k < cols; k++) {
                acc += data_self[k] * other.data[k*other.cols+j];
            }
            nu.data[i*nu.cols+j] = acc;
        }
    } 

    // We can transpose the other matrix to improve locality
    // ~0.16µs
#elif MUL_METHOD == 1
    const Matrix t_other = other.transpose();
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {

            double acc = 0.;
            const double *const data_self = &data[i*cols];
            const double *const data_other = &t_other.data[j*t_other.cols];
            
            // Pragma is a bit useless
            // #pragma omp simd reduction(+:acc) num_threads(12)
            for (int k = 0; k < cols; k++) {
                acc += data_self[k] * data_other[k];
            }
            nu.data[i*nu.cols+j] = acc;
        }
    }  

    // ikj loop ~0.16µs
#elif MUL_METHOD == 2
    for (int i = 0; i < rows; i++) {

        double *const data_nu = &nu.data[i*nu.cols];
        
        for (int k = 0; k < cols; k++) {

            const double val = data[i*cols+k];
            const double *const data_other = &other.data[k*other.cols];
            
            #pragma omp simd reduction(+:data_nu[:other.cols])
            for (int j = 0; j < other.cols; j++) {
                data_nu[j] += val * data_other[j];
            }
        }
    }

    // kij loop ~0.2µs
#elif MUL_METHOD == 3
    for (int k = 0; k < cols; k++) {
        
        const double *const data_other = &other.data[k*other.cols];
        
        for (int i = 0; i < rows; i++) {

            const double val = data[i*cols+k];
            double *const data_nu = &nu.data[i*nu.cols];
            
            #pragma omp simd reduction(+:data_nu[:other.cols])
            for (int j = 0; j < other.cols; j++) {
                data_nu[j] += val * data_other[j];
            }
        }
    }

    // Tiling
    // ~2.5µs
#elif MUL_METHOD == 4
    const int tile_size = 64;
    for (int i = 0; i < rows; i += tile_size) {
        for (int j = 0; j < other.cols; j += tile_size) {
            for (int k = 0; k < cols; k += tile_size) {

                const int i_max = std::min(i + tile_size, rows);
                const int j_max = std::min(j + tile_size, other.cols);
                const int k_max = std::min(k + tile_size, cols);

                for (int ii = i; ii < i_max; ii++) {
                    for (int jj = j; jj < j_max; jj++) {

                        double acc = 0.;
                        const double *const data_self = &data[ii*cols];
                        
                        #pragma omp simd reduction(+:acc)
                        for (int kk = k; kk < k_max; kk++) {
                            acc += data_self[kk] * other.data[kk*other.cols+jj];
                        }
                        nu.data[ii*nu.cols+jj] += acc;
                    }
                }
            }
        }
    }

#else 
    throw std::invalid_argument("Invalid multiplication method");

#endif
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

#define TRANSPOSE_METHOD 1
Matrix Matrix::transpose() const
{
    Matrix nu(cols, rows);
    
#if TRANSPOSE_METHOD == 0
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            nu.data[j*nu.cols+i] = data[i*cols+j];
        }
    } 

    // Block transpose
#elif TRANSPOSE_METHOD == 1
    const int block_size = 64;
    for (int i = 0; i < rows; i += block_size) {
        for (int j = 0; j < cols; j += block_size) {

            const int i_max = std::min(i + block_size, rows);
            const int j_max = std::min(j + block_size, cols);

            for (int ii = i; ii < i_max; ii++) {
                for (int jj = j; jj < j_max; jj++) {
                    nu.data[jj*nu.cols+ii] = data[ii*cols+jj];
                }
            }
        }
    }

#endif

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
