#pragma once

__global__ void Division(double* pMatrix, std::size_t pNumElements, std::size_t pRow);

__global__ void PostDivision(double* pMatrix, const double* pEqualities, double* pResults, std::size_t pNumElements, std::size_t pRow);

__global__ void Elimination(double* pMatrix, double* pEqualities, const double* pResults, std::size_t pNumElements, std::size_t pRow, std::size_t pMemoryOffset);

__global__ void EliminationTwo(double* pMatrix, double* pResults, std::size_t pNumElements, std::size_t pRow, std::size_t pMemoryOffset);