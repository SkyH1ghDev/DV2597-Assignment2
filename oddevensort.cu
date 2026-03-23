#include <cstdint>

#include "oddevensort.cuh"

__global__ void OneBlockSort(int* data, std::size_t dataSize)
{
    const std::uint32_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j = 0; j < dataSize; ++j)
    {
        __syncthreads();

        std::uint32_t i = 2 * baseIndex + 1;

        while (i < dataSize)
        {
            if (data[i - 1] > data[i])
            {
                std::swap(data[i], data[i - 1]);
            }

            i += 2 * blockDim.x;
        }

        __syncthreads();

        i = 2 * baseIndex + 1;

        while (i + 1 < dataSize)
        {
            if (data[i] > data[i + 1])
            {
                std::swap(data[i], data[i + 1]);
            }

            i += 2 * blockDim.x;
        }
    }
}

__global__ void MultiBlockSort_1(int* data, std::size_t dataSize)
{
    const std::uint32_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j = 0; j < dataSize; ++j)
    {
        __syncthreads();

        std::uint32_t i = 2 * baseIndex + 1;

        while (i < dataSize)
        {
            if (data[i - 1] > data[i])
            {
                std::swap(data[i], data[i - 1]);
            }

            i += 2 * blockDim.x;
        }

        __syncthreads();

        i = 2 * baseIndex + 1;

        while (i + 1 < dataSize)
        {
            if (data[i] > data[i + 1])
            {
                std::swap(data[i], data[i + 1]);
            }

            i += 2 * blockDim.x;
        }
    }
    /*const std::uint64_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint64_t i = 2 * baseIndex + 1;

    if (i >= dataSize)
    {
        return;
    }

    if (data[i] < data[i - 1])
    {
        std::swap(data[i], data[i - 1]);
    }*/
}

__global__ void MultiBlockSort_2(int* data, std::size_t dataSize)
{
    const std::uint64_t baseIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const std::uint64_t i = 2 * baseIndex + 1;

    if (i + 1 >= dataSize)
    {
        return;
    }

    if (data[i] > data[i + 1])
    {
        std::swap(data[i], data[i + 1]);
    }
}