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
    // TODO
    return 0.0;
}

void DistributedMatrix::set(int i, int j, double value)
{
    // TODO
}

int DistributedMatrix::globalColIndex(int localColIdx) const
{
    // TODO
    return -1;
}

int DistributedMatrix::localColIndex(int globalColIdx) const
{
    // TODO
    return -1;
}

int DistributedMatrix::ownerProcess(int globalColIdx) const
{
    // TODO
    return -1;
}

void DistributedMatrix::fill(double value)
{
    // TODO
}

DistributedMatrix DistributedMatrix::operator+(const DistributedMatrix& other) const
{
    // TODO
    return DistributedMatrix(*this);
}

DistributedMatrix DistributedMatrix::operator-(const DistributedMatrix& other) const
{
    // TODO
    return DistributedMatrix(*this);
}

DistributedMatrix DistributedMatrix::operator*(double scalar) const
{
    // TODO
    return DistributedMatrix(*this);
}

Matrix DistributedMatrix::transpose() const
{
    // TODO
    return Matrix(globalCols, globalRows);
}

void DistributedMatrix::sub_mul(double scalar, const DistributedMatrix& other)
{
    // TODO
}

DistributedMatrix DistributedMatrix::apply(const std::function<double(double)>& func) const
{
    // TODO
    return DistributedMatrix(*this);
}

DistributedMatrix DistributedMatrix::applyBinary(
    const DistributedMatrix& a,
    const DistributedMatrix& b,
    const std::function<double(double, double)>& func)
{
    // TODO
    return DistributedMatrix(a);
}

DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right)
{
    // TODO
    return DistributedMatrix(right);
}

Matrix DistributedMatrix::multiplyTransposed(const DistributedMatrix& other) const
{
    // TODO
    return Matrix(globalRows, other.globalRows);
}

double DistributedMatrix::sum() const
{
    // TODO
    return 0.0;
}

Matrix DistributedMatrix::gather() const
{
    int sendcount = localCols * globalRows;
    int recvcount = globalCols * globalRows;
    std::vector<double> sendbuf(sendcount), recvbuf;
    Matrix gathered(globalRows, globalCols);
    
    for (int i = 0; i < globalRows; i++) {
        for (int j = 0; j < localCols; j++) {
            sendbuf[i*localCols+j] = localData.get(i, j);
        }
    }
    
    gathered = Matrix(globalRows, globalCols);
    recvbuf.resize(recvcount);

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
