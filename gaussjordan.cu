#include "gaussjordan.cuh"

__global__ void Division(double* pMatrix, const std::size_t pNumElements, const std::size_t pRow)
{
    const std::size_t baseIndex {blockDim.x * blockIdx.x + threadIdx.x + pRow + 1};

    if (baseIndex >= pNumElements)
    {
        return;
    }

    pMatrix[pRow * pNumElements + baseIndex] = pMatrix[pRow * pNumElements + baseIndex] / pMatrix[pRow * pNumElements + pRow];
}

__global__ void PostDivision(double* pMatrix, const double* pEqualities, double* pResults, const std::size_t pNumElements, const std::size_t pRow)
{
    const std::size_t baseIndex {blockDim.x * blockIdx.x + threadIdx.x};
    if (baseIndex == 0)
    {
        pResults[pRow] = pEqualities[pRow] / pMatrix[pRow * pNumElements + pRow];
        pMatrix[pRow * pNumElements + pRow] = 1.0;
    }
}

__global__ void Elimination(double* pMatrix, double* pEqualities, const double* pResults, const std::size_t pNumElements, const std::size_t pRow, const std::size_t pMemoryOffset)
{
    const std::size_t baseIndex {blockDim.x * blockIdx.x + threadIdx.x};
    const std::size_t rowAdjustedIndex {baseIndex + pRow + 1};

    // Init Shared Memory

    extern __shared__ double sharedMemory[];
    double* a {&sharedMemory[0]};
    double* b {&sharedMemory[pNumElements]};

    for (std::size_t i {threadIdx.x}; i < pNumElements; i += blockDim.x)
    {
        a[i] = pMatrix[pRow * pNumElements + i];
    }

    // Guard Clause

    if (rowAdjustedIndex >= pNumElements)
    {
        return;
    }

    b[threadIdx.x] = pMatrix[rowAdjustedIndex * pNumElements + pRow];

    __syncthreads();

    // Do the thing

    for (std::size_t i {pRow + 1}; i < pNumElements; ++i)
    {
        pMatrix[rowAdjustedIndex * pNumElements + i] = pMatrix[rowAdjustedIndex * pNumElements + i] - b[threadIdx.x] * a[i];
    }

    pEqualities[rowAdjustedIndex] = pEqualities[rowAdjustedIndex] - b[threadIdx.x] * pResults[pRow];
    pMatrix[rowAdjustedIndex * pNumElements + pRow] = 0.0;
}

__global__ void EliminationTwo(double* pMatrix, double* pResults, const std::size_t pNumElements, const std::size_t pRow, const std::size_t pMemoryOffset)
{
    const std::size_t baseIndex {blockDim.x * blockIdx.x + threadIdx.x};

    // Init Shared Memory

    extern __shared__ double sharedMemory[];
    double* a {&sharedMemory[0]};
    double* b {&sharedMemory[pMemoryOffset / sizeof(double)]};

    for (std::size_t i {threadIdx.x}; i < pNumElements; i += blockDim.x)
    {
        a[i] = pMatrix[pRow * pNumElements + i];
    }

    // Guard Clause

    if (baseIndex >= pRow)
    {
        return;
    }

    b[threadIdx.x] = pMatrix[baseIndex * pNumElements + pRow];

    __syncthreads();

    // Do the thing

    for (std::size_t i {pRow + 1}; i < pNumElements; ++i)
    {
        pMatrix[baseIndex * pNumElements + i] = pMatrix[baseIndex * pNumElements + i] - b[threadIdx.x] * a[i];
    }

    pResults[baseIndex] = pResults[baseIndex] - b[threadIdx.x] * pResults[pRow];
    pMatrix[baseIndex * pNumElements + pRow] = 0.0;
}
