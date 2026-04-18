// Just for ease of finding the kernels' signatures and implementations
// TODO: Not for project but would be nice to extract automatically those kernels in separate source strings

__kernel void fill(__global float *matrix, float value, int rows, int cols) {
    int i = get_global_id(0);
    if (i < rows * cols) {
        matrix[i] = value;
    }
}

__kernel void add(__global const float *A, __global const float *B, __global float *C, int rows, int cols) {
    int i = get_global_id(0);
    if (i < rows * cols) {
        C[i] = A[i] + B[i];
    }
}

__kernel void sub_mul(__global float *A, __global const float *B, float scalar, int rows, int cols) {
    int i = get_global_id(0);
    if (i < rows * cols) {
        A[i] -= B[i] * scalar;
    }
}

__kernel void transpose(__global const float *A, __global float *B, int A_rows, int A_cols) {
    // r' = i % A_rows
    // c' = i / A_rows
    // i' = r' * A_cols + c' = (i % A_rows) * A_cols + i / A_rows

    int i = get_global_id(0);
    if (i < A_rows * A_cols) {
        B[i] = A[(i % A_rows) * A_cols + i / A_rows];
    }
}

__kernel void matrix_mul(__global const float* A, __global const float* B, __global float* C, int A_rows, int A_cols, int B_cols) {
    int i = get_global_id(0);
    if (i < A_rows * B_cols) {
        C[i] = 0.0f;
        int r = i / B_cols, c = i % B_cols;
        for (int k = 0; k < A_cols; k++) {
            C[i] += A[r * A_cols + k] * B[k * B_cols + c];
        }
    }
}

/**
        C

 TILE 0
vvvvvvvv
w00 w01 | w10 w11
w02 w03 | w12 w13
-----------------
w20 w21 | w30 w31
w22 w23 | w32 w33
 */
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