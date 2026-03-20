#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort
void oddeven_sort(std::vector<int>& numbers)
{
    int newPercentage = 0;
    int oldPercentage = 0;

    auto s = numbers.size();
    for (int i = 1; i <= s; i++) {

        newPercentage = (i * 100) / s;
        if (newPercentage != oldPercentage)
        {
            oldPercentage = newPercentage;
            std::cout << "Done: " << newPercentage << "%\n";
        }

        for (int j = i % 2; j < s-1; j = j + 2) {
            if (numbers[j] > numbers[j + 1]) {
                std::swap(numbers[j], numbers[j + 1]);
            }
        }
    }
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main()
{
    constexpr unsigned int size = 1 << 19; // Number of elements in the input

    srand(time(nullptr));

    for (int i{0}; i < 5; i++)
    {
        // Initialize a vector with integers of value 0
        std::vector<int> numbers(size);

        // Populate our vector with (pseudo)random numbers
        std::ranges::generate(numbers, rand);
        print_sort_status(numbers);

        auto start = std::chrono::steady_clock::now();
        oddeven_sort(numbers);
        auto end = std::chrono::steady_clock::now();

        print_sort_status(numbers);
        std::cout << std::format("Iteration: {}, Time elapsed: {}s\n", i, std::chrono::duration<double>(end - start).count());
    }
}