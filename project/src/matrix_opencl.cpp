#include "matrix_opencl.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

std::shared_ptr<KernelCache> MatrixCL::kernels_ = nullptr;

cl::Program loadAndBuildProgram(cl::Context context,
                                const std::vector<cl::Device>& devices,
                                const std::string& sourceCode,
                                const std::string& kernel_name_for_error)
{
    cl::Program program(context, sourceCode);
    try {
        program.build(devices);
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error for kernel source '" << kernel_name_for_error << "':\n"
                  << err.what() << "(" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        throw;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during program build for '" << kernel_name_for_error << "': "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
    return program;
}

// --- OpenCL Kernel Source Code ---

const std::string kernel_source_fill = R"(
    __kernel void fill(__global float* matrix, float value, int rows, int cols) {
        int i = get_global_id(0);
        if (i < rows * cols) {
            matrix[i] = value;
        }
    }
)";

const std::string kernel_source_add = R"(
    __kernel void add(__global const float* A,
                      __global const float* B,
                      __global float* C,
                      int rows, int cols) {
        int i = get_global_id(0);
        if (i < rows * cols) {
            C[i] = A[i] + B[i];
        }
    }
)";

const std::string kernel_source_sub_mul = R"(
    __kernel void sub_mul(__global float* A,
                          __global const float* B,
                          float scalar,
                          int rows, int cols) {
        int i = get_global_id(0);
        if (i < rows * cols) {
            A[i] -= B[i] * scalar;
        }
    }
)";

const std::string kernel_source_transpose = R"(
    __kernel void transpose(__global const float* A,
                            __global float* B,
                            int A_rows, int A_cols) {
        // r' = i % A_rows
        // c' = i / A_rows
        // i' = r' * A_cols + c' = (i % A_rows) * A_cols + i / A_rows

        int i = get_global_id(0);
        if (i < A_rows * A_cols) {
            B[i] = A[(i % A_rows) * A_cols + i / A_rows];
        }
    }
)";

const std::string kernel_source_matrix_mul = R"(
    __kernel void matrix_mul(__global const float* A,
                             __global const float* B,
                             __global float* C,
                             int A_rows, int A_cols, int B_cols) {
        int i = get_global_id(0);
        if (i < A_rows * B_cols) {
            C[i] = 0.0f;
            int r = i / B_cols, c = i % B_cols;
            for (int k = 0; k < A_cols; k++) {
                C[i] += A[r * A_cols + k] * B[k * B_cols + c];
            }
        }
    }
)";

const std::string kernel_source_matrix_mul_tiled = R"(
    __kernel void matrix_mul_tiled(__global const float* A, __global const float* B, __global float* C, int A_rows, int A_cols, int B_cols) {
        const int TILE = 16;
        __local float tileA[TILE][TILE];
        __local float tileB[TILE][TILE];

        int row = get_global_id(1);
        int col = get_global_id(0);
        int lrow = get_local_id(1);
        int lcol = get_local_id(0);

        float acc = 0.0f;
        for (int chunk = 0; chunk < (int)ceil((float)A_cols / TILE); chunk++) {
            tileA[lrow][lcol] = 0.0f;
            tileB[lrow][lcol] = 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (row < A_rows && chunk * TILE + lcol < A_cols)
                tileA[lrow][lcol] = A[row * A_cols + chunk * TILE + lcol];
            if (col < B_cols && chunk * TILE + lrow < A_cols)
                tileB[lrow][lcol] = B[(chunk * TILE + lrow) * B_cols + col];
            
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < TILE; k++) {
                acc += tileA[lrow][k] * tileB[k][lcol];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row < A_rows && col < B_cols)
            C[row * B_cols + col] = acc;
    }
)";

// --- KernelCache ---

void KernelCache::compileKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    if (initialized) return;

    std::cout << "Compiling OpenCL kernels..." << std::endl;
    try {
        cl::Program prog_fill = loadAndBuildProgram(context, devices, kernel_source_fill, "fill");
        kernel_fill = cl::Kernel(prog_fill, "fill");

        cl::Program prog_add = loadAndBuildProgram(context, devices, kernel_source_add, "add");
        kernel_add = cl::Kernel(prog_add, "add");

        cl::Program prog_sub_mul = loadAndBuildProgram(context, devices, kernel_source_sub_mul, "sub_mul");
        kernel_sub_mul = cl::Kernel(prog_sub_mul, "sub_mul");

        cl::Program prog_transpose = loadAndBuildProgram(context, devices, kernel_source_transpose, "transpose");
        kernel_transpose = cl::Kernel(prog_transpose, "transpose");

        cl::Program prog_matrix_mul = loadAndBuildProgram(context, devices, kernel_source_matrix_mul, "matrix_mul");
        kernel_matrix_mul = cl::Kernel(prog_matrix_mul, "matrix_mul");

        cl::Program prog_matrix_mul_tiled = loadAndBuildProgram(context, devices, kernel_source_matrix_mul_tiled, "matrix_mul_tiled");
        kernel_matrix_mul_tiled = cl::Kernel(prog_matrix_mul_tiled, "matrix_mul_tiled");

        initialized = true;
        std::cout << "OpenCL kernels compiled successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to compile one or more OpenCL kernels. Aborting." << std::endl;
        throw;
    }
}

// --- MatrixCL Static Methods ---

void MatrixCL::initializeKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    try {
        if (!kernels_ || !kernels_->initialized) {
            std::cout << "Creating and compiling kernels..." << std::endl;
            kernels_ = std::make_shared<KernelCache>();
            kernels_->compileKernels(context, devices);
        }
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in kernel initialization: "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Exception in kernel initialization: " << e.what() << std::endl;
        throw;
    }
}

// --- MatrixCL Implementation ---

size_t MatrixCL::buffer_size_bytes() const {
    return static_cast<size_t>(rows_) * cols_ * sizeof(float);
}

MatrixCL::MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data)
    : rows_(rows), cols_(cols), context_(context), queue_(queue)
{
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimension can't be negative");
    }

    size_t N = rows * cols;

    if (initial_data) {
        if (initial_data->size() != N) {
            throw std::invalid_argument("Dimension mismatch between arguments");
        }
        
        buffer_ = cl::Buffer(
            context_,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            buffer_size_bytes(),
            const_cast<float*>(initial_data->data())
        );

    } else {
        buffer_ = cl::Buffer(
            context_,
            CL_MEM_READ_WRITE,
            buffer_size_bytes()
        );
    }
}

MatrixCL::MatrixCL(const MatrixCL& other)
    : rows_(other.rows_), cols_(other.cols_),
      context_(other.context_), queue_(other.queue_)
{
    buffer_ = cl::Buffer(
        context_,
        CL_MEM_READ_WRITE,
        buffer_size_bytes()
    );

    queue_.enqueueCopyBuffer(
        other.buffer_,
        buffer_,
        0,
        0,
        buffer_size_bytes()
    );

    queue_.finish();
}

MatrixCL& MatrixCL::operator=(const MatrixCL& other)
{
    if (this == &other) return *this;

    rows_ = other.rows_;
    cols_ = other.cols_;
    context_ = other.context_;
    queue_ = other.queue_;

    buffer_ = cl::Buffer(
        context_,
        CL_MEM_READ_WRITE,
        buffer_size_bytes()
    );

    queue_.enqueueCopyBuffer(
        other.buffer_,
        buffer_,
        0,
        0,
        buffer_size_bytes()
    );

    queue_.finish();

    return *this;
}

int MatrixCL::numRows() const { return rows_; }
int MatrixCL::numCols() const { return cols_; }
cl::Context MatrixCL::getContext() const { return context_; }
cl::CommandQueue MatrixCL::getQueue() const { return queue_; }
const cl::Buffer& MatrixCL::getBuffer() const { return buffer_; }

std::vector<float> MatrixCL::copyToHost() const
{
    std::vector<float> host_data(static_cast<size_t>(rows_) * cols_);
    size_t size = buffer_size_bytes();
    if (size == 0) return host_data;

    queue_.enqueueReadBuffer(
        buffer_,
        CL_TRUE,
        0,
        buffer_size_bytes(),
        host_data.data()        
    );

    return host_data;
}

// Helpful macros
#define nqfin(q_, kernel_, N_) do { \
    (q_).enqueueNDRangeKernel((kernel_), cl::NullRange, cl::NDRange(N_)); \
    (q_).finish(); \
} while(0);

#define setargs(kernel_, ...) do { \
    int arg_idx = 0; \
    auto set_arg_lambda = [&](auto&&... args) { \
        (((kernel_).setArg(arg_idx++, args)), ...); \
    }; \
    set_arg_lambda(__VA_ARGS__); \
} while(0);

void MatrixCL::fill(float value)
{
    size_t N = rows_ * cols_; 
    if (N == 0) return;

    cl::Kernel kernel = kernels_->kernel_fill;
    setargs(kernel, buffer_, value, rows_, cols_);
    nqfin(queue_, kernel, N);
}

MatrixCL MatrixCL::operator+(const MatrixCL& other) const
{
    MatrixCL result(rows_, cols_, context_, queue_);
    size_t N = rows_ * cols_; 
    if (N == 0) return result;

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Dimension mismatch during addition");
    } 

    cl::Kernel kernel = kernels_->kernel_add;
    setargs(kernel, buffer_, other.buffer_, result.buffer_, rows_, cols_);
    nqfin(queue_, kernel, N);

    return result;
}

MatrixCL MatrixCL::operator-(const MatrixCL& other) const
{
    MatrixCL result(*this);
    size_t N = rows_ * cols_; 
    if (N == 0) return result;

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Dimension mismatch during subtraction");
    }
    
    cl::Kernel kernel = kernels_->kernel_sub_mul;
    setargs(kernel, result.buffer_, other.buffer_, 1.0f, rows_, cols_);
    nqfin(queue_, kernel, N);

    return result;
}

MatrixCL MatrixCL::operator*(float scalar) const
{
    MatrixCL result(rows_, cols_, context_, queue_);
    size_t N = rows_ * cols_; 
    if (N == 0) return result;

    cl::Kernel kernel;

    // result := this * scalar <=> result := 0 then result := result - this * (-scalar)
    kernel = kernels_->kernel_fill;
    setargs(kernel, result.buffer_, 0.0f, rows_, cols_);
    nqfin(queue_, kernel, N);

    kernel = kernels_->kernel_sub_mul;
    setargs(kernel, result.buffer_, buffer_, -scalar, rows_, cols_);
    nqfin(queue_, kernel, N);

    return result;
}

#ifndef CL_MUL_METHOD
#define CL_MUL_METHOD 1
#endif
MatrixCL MatrixCL::operator*(const MatrixCL& other) const
{
    int C_rows = this->rows_;
    int C_cols = other.cols_;
    MatrixCL result(C_rows, C_cols, context_, queue_);
    size_t N = C_rows * C_cols; 
    if (N == 0) return result;

    if (cols_ != other.rows_) {
        throw std::invalid_argument("Dimension mismatch during matrix multiplication");
    }

#if CL_MUL_METHOD == 0
    cl::Kernel kernel = kernels_->kernel_matrix_mul;
    setargs(kernel, buffer_, other.buffer_, result.buffer_, rows_, cols_, other.cols_);
    nqfin(queue_, kernel, N); 
#elif CL_MUL_METHOD == 1

#define ceil_div(a, b) (((a) + (b) - 1) / (b))

    cl::Kernel kernel = kernels_->kernel_matrix_mul_tiled;
    setargs(kernel, buffer_, other.buffer_, result.buffer_, rows_, cols_, other.cols_);

    const int TILE = 16; 
    cl::NDRange global(ceil_div(other.cols_, TILE)*TILE, ceil_div(rows_, TILE)*TILE);
    cl::NDRange local(TILE, TILE);

    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    queue_.finish();

#else
    throw std::invalid_argument("Invalid multiplication method");
#endif

    return result;
}

MatrixCL MatrixCL::transpose() const
{
    MatrixCL result(cols_, rows_, context_, queue_);
    size_t N = rows_ * cols_; 
    if (N == 0) return result;

    cl::Kernel kernel = kernels_->kernel_transpose;
    setargs(kernel, buffer_, result.buffer_, rows_, cols_);
    nqfin(queue_, kernel, N);

    return result;
}

void MatrixCL::sub_mul(float scalar, const MatrixCL& other)
{
    size_t N = rows_ * cols_; 
    if (N == 0) return;

    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Dimension mismatch during sub-mul");
    }

    cl::Kernel kernel = kernels_->kernel_sub_mul;
    setargs(kernel, buffer_, other.buffer_, scalar, rows_, cols_);
    nqfin(queue_, kernel, N);
}
