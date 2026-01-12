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

__global__ void Elimination(double* pMatrix, double* pEqualities, double* pResults, std::size_t pNumElements, std::size_t pRow)
{
    const std::size_t baseIndex {blockDim.x * blockIdx.x + threadIdx.x + pRow + 1};

    if (baseIndex >= pNumElements)
    {
        return;
    }

    for (std::size_t i {pRow + 1}; i < pNumElements; ++i)
    {
        pMatrix[baseIndex * pNumElements + i] = pMatrix[baseIndex * pNumElements + i] - pMatrix[baseIndex * pNumElements + pRow] * pMatrix[pRow * pNumElements + i];
    }

    pEqualities[baseIndex] = pEqualities[baseIndex] - pMatrix[baseIndex * pNumElements + pRow] * pResults[pRow];
    pMatrix[baseIndex * pNumElements + pRow] = 0.0;
}

__global__ void EliminationTwo(double* pMatrix, double* pEqualities, double* pResults, std::size_t pNumElements, std::size_t pRow)
{
    const std::size_t baseIndex {blockDim.x * blockIdx.x + threadIdx.x};

    if (baseIndex >= pRow)
    {
        return;
    }

    for (std::size_t i {pRow + 1}; i < pNumElements; ++i)
    {
        pMatrix[baseIndex * pNumElements + i] = pMatrix[baseIndex * pNumElements + i] - pMatrix[baseIndex * pNumElements + pRow] * pMatrix[pRow * pNumElements + i];
    }

    pResults[baseIndex] = pResults[baseIndex] - pMatrix[baseIndex * pNumElements + pRow] * pResults[pRow];
    pMatrix[baseIndex * pNumElements + pRow] = 0.0;
}
