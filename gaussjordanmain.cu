#include "gaussjordan.cuh"

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

constexpr std::size_t MAX_SIZE {4096};

__host__ void Print_Matrix(const std::array<double, MAX_SIZE * MAX_SIZE>& pMatrix, const std::array<double, MAX_SIZE>& pResults, std::size_t pNumElements);
__host__ void Init_Matrix(std::array<double, MAX_SIZE * MAX_SIZE>& pMatrix, std::array<double, MAX_SIZE>& pEqualities, std::array<double, MAX_SIZE>& pResults, std::string_view pInitType, std::size_t pNumElements, std::uint64_t pMaxElementValue, bool pPrintFlag);
__host__ int Read_Options(int argc, char** argv, std::string& pInitType, std::size_t& pNumElements, std::uint64_t& pMaxElementValue, bool& pPrintFlag);

__host__ int Read_Options(int argc, char** argv, std::string& pInitType, std::size_t& pNumElements, std::uint64_t& pMaxElementValue, bool& pPrintFlag)
{
    char* prog;

    prog = *argv;
    while (++argv, --argc > 0)
    {
        if (**argv == '-')
        {
            switch (*++ * argv) {
                case 'n':
                    --argc;
                    pNumElements = atol(*++argv);
                    break;
                case 'h':
                    printf("\nHELP: try sor -u \n\n");
                    exit(0);
                case 'u':
                    printf("\nUsage: gaussian [-n problemsize]\n");
                    printf("           [-D] show default values \n");
                    printf("           [-h] help \n");
                    printf("           [-I init_type] fast/rand \n");
                    printf("           [-m maxnum] max random no \n");
                    printf("           [-P print_switch] 0/1 \n");
                    exit(0);
                case 'D':
                    printf("\nDefault:  n         = %lu ", pNumElements);
                    printf("\n          Init      = rand");
                    printf("\n          maxnum    = 5 ");
                    printf("\n          P         = 0 \n\n");
                    exit(0);
                case 'I':
                    --argc;
                    pInitType = *++argv;
                    break;
                case 'm':
                    --argc;
                    pMaxElementValue = atol(*++argv);
                    break;
                case 'P':
                    --argc;
                    pPrintFlag = atoi(*++argv);
                    break;
                default:
                    printf("%s: ignored option: -%s\n", prog, *argv);
                    printf("HELP: try %s -u \n\n", prog);
                    break;
            }
        }
    }

    return 0;
}

__host__ void Init_Matrix(std::array<double, MAX_SIZE * MAX_SIZE>& pMatrix, std::array<double, MAX_SIZE>& pEqualities, std::array<double, MAX_SIZE>& pResults, const std::string_view pInitType, const std::size_t pNumElements, const std::uint64_t pMaxElementValue, const bool pPrintFlag)
{
    int i, j;

    printf("\nsize      = %lux%lu ", pNumElements, pNumElements);
    printf("\nmaxnum    = %lu \n", pMaxElementValue);
    printf("Init	  = %s \n", pInitType.data());
    printf("Initializing matrix...");

    if (strcmp(pInitType.data(), "rand") == 0) {
        for (i = 0; i < pNumElements; i++) {
            for (j = 0; j < pNumElements; j++) {
                if (i == j) /* diagonal dominance */
                    pMatrix[i * pNumElements + j] = static_cast<double>(rand() % pMaxElementValue) + 5.0;
                else
                    pMatrix[i * pNumElements + j] = static_cast<double>(rand() % pMaxElementValue) + 1.0;
            }
        }
    }
    if (strcmp(pInitType.data(), "fast") == 0) {
        for (i = 0; i < pNumElements; i++) {
            for (j = 0; j < pNumElements; j++) {
                if (i == j) /* diagonal dominance */
                    pMatrix[i * pNumElements + j] = 5.0;
                else
                    pMatrix[i * pNumElements + j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < pNumElements; i++) {
        pEqualities[i] = 2.0;
        pResults[i] = 1.0;
    }

    printf("done \n\n");
    if (pPrintFlag)
        Print_Matrix(pMatrix, pResults, pNumElements);
}

__host__ void Print_Matrix(const std::array<double, MAX_SIZE * MAX_SIZE>& pMatrix, const std::array<double, MAX_SIZE>& pResults, const std::size_t pNumElements)
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < pNumElements; i++) {
        printf("[");
        for (j = 0; j < pNumElements; j++)
            printf(" %5.2f,", pMatrix[i * pNumElements + j]);
        printf("]\n");
    }
    printf("Vector y:\n[");
    for (j = 0; j < pNumElements; j++)
        printf(" %5.2f,", pResults[j]);
    printf("]\n");
    printf("\n\n");
}

__host__ int main(int argc, char** argv)
{
    static std::array<double, MAX_SIZE * MAX_SIZE> matrix {0};
    static std::array<double, MAX_SIZE> equalities {0};
    static std::array<double, MAX_SIZE> results {0};

    std::size_t numElements {2048};
    std::uint64_t maxElementValue {15};
    std::string initType {"fast"};
    bool printFlag {false};

    printf("Gauss Jordan\n");

    Read_Options(argc, argv, initType, numElements, maxElementValue, printFlag);
    Init_Matrix(matrix, equalities, results, initType, numElements, maxElementValue, printFlag);

    double* cudaMatrix = nullptr;
    std::size_t cudaMatrixSize = sizeof(*cudaMatrix) * numElements * numElements;

    double* cudaEqualities = nullptr;
    std::size_t cudaEqualitiesSize = sizeof(*cudaEqualities) * numElements;

    double* cudaResults = nullptr;
    std::size_t cudaResultsSize = sizeof(*cudaResults) * numElements;

    cudaMalloc(reinterpret_cast<void**>(&cudaMatrix), cudaMatrixSize);
    cudaMalloc(reinterpret_cast<void**>(&cudaEqualities), cudaEqualitiesSize);
    cudaMalloc(reinterpret_cast<void**>(&cudaResults), cudaResultsSize);

    cudaMemcpy(cudaMatrix, matrix.data(), cudaMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaEqualities, equalities.data(), cudaEqualitiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaResults, results.data(), cudaResultsSize, cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();

    for (std::size_t row {0}; row < numElements; ++row)
    {
        constexpr std::uint32_t numBlocks {32};
        constexpr std::uint32_t numThreads {64};

        Division<<<numBlocks, numThreads>>>(cudaMatrix, numElements, row);
        PostDivision<<<1, 1>>>(cudaMatrix, cudaEqualities, cudaResults, numElements, row);
        Elimination<<<numBlocks, numThreads>>>(cudaMatrix, cudaEqualities, cudaResults, numElements, row);
        EliminationTwo<<<numBlocks, numThreads>>>(cudaMatrix, cudaEqualities, cudaResults, numElements, row);
    }

    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(matrix.data(), cudaMatrix, cudaMatrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(equalities.data(), cudaEqualities, cudaEqualitiesSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(results.data(), cudaResults, cudaResultsSize, cudaMemcpyDeviceToHost);

    cudaFree(cudaResults);
    cudaFree(cudaEqualities);
    cudaFree(cudaMatrix);

    if (printFlag)
    {
        Print_Matrix(matrix, results, numElements);
    }

    std::cout << "Elapsed time = " << std::chrono::duration<double> {end - start}.count() << " sec\n";
}