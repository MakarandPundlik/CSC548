#include <iostream>
#include <vector>
#include <random>
#include <limits.h>
#include <chrono>
#include <algorithm>
#include <cstdlib> // For atoi, atof
#include <omp.h>   // For OpenMP

std::vector<std::vector<int>> generateRandomMatrix(int n, double density)
{
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    std::uniform_int_distribution<> weightDistrib(1, 10); // Weights between 1 and 10

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                matrix[i][j] = 0;
            }
            else if (distrib(gen) < density)
            {
                matrix[i][j] = weightDistrib(gen);
            }
            else
            {
                matrix[i][j] = INT_MAX;
            }
        }
    }
    return matrix;
}

void floydWarshallOMP(std::vector<std::vector<int>> &graph)
{
    int n = graph.size();
    for (int k = 0; k < n; ++k)
    {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (graph[i][k] != INT_MAX && graph[k][j] != INT_MAX)
                {
                    if (graph[i][j] > graph[i][k] + graph[k][j])
                    {
                        graph[i][j] = graph[i][k] + graph[k][j];
                    }
                }
            }
        }
    }
}

bool compareResults(const std::vector<std::vector<int>> &result1, const std::vector<std::vector<int>> &result2, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (result1[i][j] != result2[i][j])
            {
                std::cerr << "Mismatch at (" << i << ", " << j << "): OMP = " << result1[i][j] << ", CPU = " << result2[i][j] << std::endl;
                return false;
            }
        }
    }
    std::cout << "Results from OpenMP and CPU match!\n";
    return true;
}

void floydWarshallCPU(std::vector<std::vector<int>> &graph)
{
    int n = graph.size();
    for (int k = 0; k < n; ++k)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (graph[i][k] != INT_MAX && graph[k][j] != INT_MAX)
                {
                    if (graph[i][j] > graph[i][k] + graph[k][j])
                    {
                        graph[i][j] = graph[i][k] + graph[k][j];
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int n = 512;
    double density = 0.3;

    std::cout << "Floyd-Warshall OpenMP/CPU Comparison\n";
    std::cout << "---------------------------------------\n";
    std::cout << "Arguments: [N (positive integer)] [Density (0.0 to 1.0)]\n";
    std::cout << "---------------------------------------\n";
    std::cout << "Limits for N: Positive integer. Recommended up to system RAM limits.\n";
    std::cout << "Ensure your system has enough RAM for the matrix (N*N*4 bytes).\n";
    std::cout << "Limits for Density: Floating-point number between 0.0 and 1.0.\n";
    std::cout << "---------------------------------------\n";

    if (argc > 1)
    {
        int parsed_n = std::atoi(argv[1]);
        if (parsed_n > 0)
        {
            n = parsed_n;
            std::cout << "Using matrix size N = " << n << ".\n";
        }
        else
        {
            std::cerr << "Error: Matrix size (N) must be a positive integer. Using default N = 512.\n";
        }
    }
    else
    {
        std::cout << "No matrix size (N) provided. Using default N = 512.\n";
    }

    if (argc > 2)
    {
        double parsed_density = std::atof(argv[2]);
        if (parsed_density >= 0.0 && parsed_density <= 1.0)
        {
            density = parsed_density;
            std::cout << "Using density = " << density << ".\n";
        }
        else
        {
            std::cerr << "Error: Density must be between 0.0 and 1.0. Using default density = 0.3.\n";
        }
    }
    else
    {
        std::cout << "No density provided. Using default density = 0.3.\n";
    }

    long long required_memory = static_cast<long long>(n) * n * sizeof(int);
    std::cout << "Estimated host memory required: " << required_memory / (1024 * 1024) << " MB.\n";
    if (n > 4096)
    {
        std::cerr << "Warning: Large matrix size. This may lead to excessive execution time.\n";
    }

    std::cout << "Generating a " << n << "x" << n << " random adjacency matrix with density " << density << "...\n";
    std::vector<std::vector<int>> graph_host = generateRandomMatrix(n, density);
    std::cout << "Matrix generated.\n";

    std::vector<std::vector<int>> graph_omp = graph_host;
    std::cout << "Running Floyd-Warshall with OpenMP...\n";
    auto start_omp = std::chrono::high_resolution_clock::now();
    floydWarshallOMP(graph_omp);
    auto end_omp = std::chrono::high_resolution_clock::now();
    auto duration_omp = std::chrono::duration_cast<std::chrono::milliseconds>(end_omp - start_omp);
    std::cout << "OpenMP Floyd-Warshall execution time: " << duration_omp.count() << " ms\n";

    std::vector<std::vector<int>> graph_cpu = graph_host;
    std::cout << "Running Floyd-Warshall on CPU...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    floydWarshallCPU(graph_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    std::cout << "CPU Floyd-Warshall execution time: " << duration_cpu.count() << " ms\n";
    bool resultsMatch = compareResults(graph_omp, graph_cpu, n);

    return 0;
}