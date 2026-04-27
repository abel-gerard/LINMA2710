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

#ifndef MUL_METHOD
#define MUL_METHOD 1
#endif
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
            
#ifdef USE_OMP
            #pragma omp simd reduction(+:acc) num_threads(12)
#endif
            for (int k = 0; k < cols; k++) {
                acc += data_self[k] * data_other[k];
            }
            nu.data[i*nu.cols+j] = acc;
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
