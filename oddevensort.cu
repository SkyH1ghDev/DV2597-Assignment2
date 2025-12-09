#include <cstdint>

#include "oddevensort.cuh"

__global__ void OneBlockSort(int* data, int dataSize)
{
    const std::uint32_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t i = 2 * baseIndex + 1;

    for (int j = 0; j < dataSize; ++j)
    {
        __syncthreads();

        if (i > dataSize)
        {
            goto sync1;
        }

        if (data[i] < data[i - 1])
        {
            std::swap(data[i], data[i - 1]);
        }

        sync1:
        __syncthreads();

        if (i > dataSize || i + 1 > dataSize)
        {
            goto sync2;
        }

        if (data[i] > data[i + 1])
        {
            std::swap(data[i], data[i + 1]);
        }

        sync2:
    }
}

__global__ void MultiBlockSort_1(int* data, int dataSize)
{
    const std::uint32_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t i = 2 * baseIndex + 1;

    if (i > dataSize)
    {
        return;
    }

    if (data[i] < data[i - 1])
    {
        std::swap(data[i], data[i - 1]);
    }
}

__global__ void MultiBlockSort_2(int* data, int dataSize)
{
    const std::uint32_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint32_t i = 2 * baseIndex + 1;

    if (i > dataSize || i + 1 > dataSize)
    {
        return;
    }

    if (data[i] > data[i + 1])
    {
        std::swap(data[i], data[i + 1]);
    }
}