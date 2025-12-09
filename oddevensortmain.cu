#include <iostream>
#include <chrono>

#include "oddevensort.cuh"


__host__ void print_sort_status(const std::vector<int>& numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") <<            std::endl;
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
    constexpr unsigned int size = 7; // Number of elements in the input

    // Initialize a vector with integers of value 0
    std::vector<int> numbers(size);

    // Populate our vector with (pseudo)random numbers
    srand(time(nullptr));
    std::generate(numbers.begin(), numbers.end(), rand);
    for (auto& v : numbers)
    {
        v = v / (std::numeric_limits<int>::max() / 20);
    }

    print_numbers(numbers);
    print_sort_status(numbers);

    int* cudaData = nullptr;
    std::size_t cudaDataSize = sizeof(*cudaData) * numbers.size();
    cudaMalloc(reinterpret_cast<void**>(&cudaData), cudaDataSize);
    cudaMemcpy(cudaData, numbers.data(), cudaDataSize, cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();
    //OneBlockSort<<<1, 5>>>(cudaData, numbers.size());

    for (int i = 0; i < cudaDataSize; ++i)
    {
        MultiBlockSort_1<<<3, 1>>>(cudaData, numbers.size());
        MultiBlockSort_2<<<3, 1>>>(cudaData, numbers.size());
    }

    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(numbers.data(), cudaData, cudaDataSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaData);

    print_numbers(numbers);
    print_sort_status(numbers);
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
}
