#include <iostream>
#include <chrono>
#include <algorithm>

#include "oddevensort.cuh"


__host__ void print_sort_status(const std::vector<int>& numbers)
{
    std::cout << "The input is sorted?: " << (std::ranges::is_sorted(numbers) == 0 ? "False" : "True") <<            std::endl;
}

__host__ void print_numbers(const std::vector<int>& numbers)
{
    for (auto v : numbers)
    {
        std::cout << v << " ";
    }
    std::cout << "\n";
}



__host__ int main()
{
    constexpr unsigned int size = 1 << 19; // Number of elements in the input

    srand(time(nullptr));
    std::vector<int> numbers(size);

    for (int i{0}; i < 25; ++i)
    {
        std::uint32_t numBlocks{4096 / static_cast<std::uint32_t>(pow(2, static_cast<std::uint32_t>(i / 5)))};
        std::uint32_t numThreads{64 * static_cast<std::uint32_t>(pow(2, static_cast<std::uint32_t>(i / 5)))};

        // Initialize a vector with integers of value 0

        // Populate our vector with (pseudo)random numbers
        std::ranges::generate(numbers, rand);
        for (auto& v : numbers)
        {
            v = v / (std::numeric_limits<int>::max() / 20);
        }

        int* cudaData = nullptr;
        std::size_t cudaDataSize = sizeof(*cudaData) * numbers.size();
        cudaMalloc(reinterpret_cast<void**>(&cudaData), cudaDataSize);
        cudaMemcpy(cudaData, numbers.data(), cudaDataSize, cudaMemcpyHostToDevice);

        auto start = std::chrono::steady_clock::now();

        // ONE KERNEL

        //OneBlockSort<<<1, 1024>>>(cudaData, numbers.size());

        // MULTIPLE KERNELS

        MultiBlockSort_1<<<1, 1024>>>(cudaData, numbers.size());

        /*for (int i = 0; i < cudaDataSize; ++i)
        {
            MultiBlockSort_1<<<numBlocks, numThreads>>>(cudaData, numbers.size());
            MultiBlockSort_2<<<numBlocks, numThreads>>>(cudaData, numbers.size());
        }*/

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();

        cudaMemcpy(numbers.data(), cudaData, cudaDataSize, cudaMemcpyDeviceToHost);
        cudaFree(cudaData);

        std::cout << std::format("Iteration: {} (numBlocks: {}, numThreads: {}), Time elapsed: {}s\n", i, numBlocks, numThreads, std::chrono::duration<double>(end - start).count());
    }
}
