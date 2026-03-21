#include "distributed_matrix.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

#if 1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter" 
#endif

// The matrix is split by columns across MPI processes.
// Each process stores a local Matrix with a subset of columns.
// Columns are distributed as evenly as possible.

DistributedMatrix::DistributedMatrix(const Matrix& matrix, int numProcs)
    : globalRows(matrix.numRows()),
      globalCols(matrix.numCols()),
      localCols(0),
      startCol(0),
      numProcesses(numProcs),
      rank(0),
      localData(matrix.numRows(), 1)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int baseSizePerProc = globalCols / numProcesses;
    int remainderSize = globalCols % numProcesses;

    localCols = baseSizePerProc + (rank < remainderSize);
    startCol = rank * baseSizePerProc + (rank < remainderSize ? rank : remainderSize);

    localData = Matrix(globalRows, localCols);
    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            localData.set(i, j, matrix.get(i, startCol + j));
        }
    }
}

DistributedMatrix::DistributedMatrix(const DistributedMatrix& other)
    : globalRows(other.globalRows),
      globalCols(other.globalCols),
      localCols(other.localCols),
      startCol(other.startCol),
      numProcesses(other.numProcesses),
      rank(other.rank),
      localData(other.localData)
{

}

int DistributedMatrix::numRows() const { return globalRows; }
int DistributedMatrix::numCols() const { return globalCols; }
const Matrix& DistributedMatrix::getLocalData() const { return localData; }

double DistributedMatrix::get(int i, int j) const
{
    int localJ = localColIndex(j);
    if (localJ < 0 || localJ >= localCols)
        throw std::out_of_range("Column index not owned by this process");

    return localData.get(i, localJ);
}

void DistributedMatrix::set(int i, int j, double value)
{
    int localJ = localColIndex(j);
    if (localJ < 0 || localJ >= localCols)
        throw std::out_of_range("Column index not owned by this process");

    localData.set(i, localJ, value);
}

int DistributedMatrix::globalColIndex(int localColIdx) const
{
    return startCol + localColIdx;
}

int DistributedMatrix::localColIndex(int globalColIdx) const
{
    return globalColIdx - startCol;
}

int DistributedMatrix::ownerProcess(int globalColIdx) const
{
    return localColIndex(globalColIdx) >= 0 && localColIndex(globalColIdx) < localCols ? rank : -1;
}

void DistributedMatrix::fill(double value)
{
    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            localData.set(i, j, value);
        }
    }
}

DistributedMatrix DistributedMatrix::operator+(const DistributedMatrix& other) const
{
    DistributedMatrix result(*this);

    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            result.localData.set(i, j, localData.get(i, j) + other.localData.get(i, j));
        }
    }

    return result;
}

DistributedMatrix DistributedMatrix::operator-(const DistributedMatrix& other) const
{
    DistributedMatrix result(*this);

    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            result.localData.set(i, j, localData.get(i, j) - other.localData.get(i, j));
        }
    }

    return result;
}

DistributedMatrix DistributedMatrix::operator*(double scalar) const
{
    DistributedMatrix result(*this);

    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            result.localData.set(i, j, localData.get(i, j) * scalar);
        }
    }

    return result;
}

Matrix DistributedMatrix::transpose() const
{
    Matrix result = gather();

    return result.transpose();
}

void DistributedMatrix::sub_mul(double scalar, const DistributedMatrix& other)
{
    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            localData.set(i, j, localData.get(i, j) - other.localData.get(i, j) * scalar);
        }
    }
}

DistributedMatrix DistributedMatrix::apply(const std::function<double(double)>& func) const
{
    DistributedMatrix result(*this);

    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            result.localData.set(i, j, func(localData.get(i, j)));
        }
    }

    return result;
}

DistributedMatrix DistributedMatrix::applyBinary(
    const DistributedMatrix& a,
    const DistributedMatrix& b,
    const std::function<double(double, double)>& func)
{
    DistributedMatrix result(a);

    for (int i = 0; i < a.globalRows; i++) {
        for (int j = 0; j < a.localCols; j++) {
            result.localData.set(i, j, func(a.localData.get(i, j), b.localData.get(i, j)));
        }
    }

    return result;
}

DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right)
{
    Matrix matResult(left.numRows(), right.localCols);

    for (int i = 0; i < matResult.numRows(); i++) {
        for (int j = 0; j < matResult.numCols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < left.numCols(); k++) {
                sum += left.get(i, k) * right.localData.get(k, j);
            }
            matResult.set(i, j, sum);
        }
    }

    // Copy back to a DistributedMatrix without partioning it again
    DistributedMatrix result(right);
    result.globalRows = left.numRows();
    result.localData = matResult;

    return result;
}

Matrix DistributedMatrix::multiplyTransposed(const DistributedMatrix& other) const
{
    // (m x n) * (p x n)^T -> (m x p)
    int resultRows = globalRows, resultCols = other.globalRows;
    int count = globalRows * other.globalRows;
    std::vector<double> vecResult(count);

    // Compute A^(i)B^(i)^T for the current rank
    for (int i = 0; i < resultRows; i++) {
        for (int j = 0; j < resultCols; j++) {
            double sum = 0.0;
            for (int k = 0; k < localCols; k++) {
                sum += localData.get(i, k) * other.localData.get(j, k);
            }
            vecResult[i * resultCols + j] = sum;
        }
    }

    // Reduce across all processes to get the final result
    MPI_Allreduce(
        MPI_IN_PLACE, vecResult.data(),
        count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    Matrix result(resultRows, resultCols);
    for (int i = 0; i < resultRows; i++) {
        for (int j = 0; j < resultCols; j++) {
            result.set(i, j, vecResult[i * resultCols + j]);
        }
    }

    return result;
}

double DistributedMatrix::sum() const
{
    // Sequential reduce + MPI_Allreduce
    double sum = 0.0;

    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            sum += localData.get(i, j);
        }
    }

    MPI_Allreduce(
        MPI_IN_PLACE, &sum,
        1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
    );

    return sum;
}

Matrix DistributedMatrix::gather() const
{
    int sendcount = localCols * globalRows;
    int recvcount = globalCols * globalRows;
    std::vector<double> sendbuf(sendcount), recvbuf(recvcount);
    Matrix gathered(globalRows, globalCols);
    
    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            sendbuf[i*localCols+j] = localData.get(i, j);
        }
    }
    
    std::vector<int> recvcounts(numProcesses), displs(numProcesses);
    int baseSizePerProc = globalCols / numProcesses;
    int remainderSize = globalCols % numProcesses;

    for (int p = 0; p < numProcesses; p++) {
        int cols = baseSizePerProc + (p < remainderSize);
        recvcounts[p] = cols * globalRows;
        displs[p] = (p > 0) ? displs[p-1] + recvcounts[p-1] : 0;
    }

    MPI_Allgatherv(
        sendbuf.data(), sendcount, MPI_DOUBLE,
        recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    // recvbuf looks like:
    // [col0 col1 col0 col1 ... col2 col3 ... colN]
    for (int p = 0; p < numProcesses; p++) {
        int localCols = recvcounts[p] / globalRows; 
        int startCol = displs[p] / globalRows;
        for (int i = 0; i < globalRows; i++) {
            for (int j = 0; j < localCols; j++) {
                gathered.set(i, startCol + j, recvbuf[displs[p] + i*localCols + j]);
            }
        }
    }

    return gathered;
}

void sync_matrix(Matrix *matrix, int rank, int src)
{
    // TODO
}
