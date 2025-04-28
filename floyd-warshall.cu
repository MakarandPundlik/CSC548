#include <iostream>
#include <vector>
#include <random>
#include <limits.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cstdlib> // For atoi, atof

std::vector<std::vector<int>> generateRandomMatrix(int n, double density)
{
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    std::uniform_int_distribution<> weightDistrib(1, 10);

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

__global__ void floydWarshallKernel(int *d, int n, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n)
    {
        if (d[i * n + k] != INT_MAX && d[k * n + j] != INT_MAX)
        {
            if (d[i * n + j] > d[i * n + k] + d[k * n + j])
            {
                d[i * n + j] = d[i * n + k] + d[k * n + j];
            }
        }
    }
}

void floydWarshallCUDA(int *d_in, int n, int *d_out)
{
    int *d_dev;
    cudaMalloc((void **)&d_dev, n * n * sizeof(int));
    cudaMemcpy(d_dev, d_in, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    for (int k = 0; k < n; ++k)
    {
        floydWarshallKernel<<<gridDim, blockDim>>>(d_dev, n, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(d_out, d_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dev);
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

bool compareResults(const std::vector<std::vector<int>> &cpuResult, const int *cudaResult, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (cpuResult[i][j] != cudaResult[i * n + j])
            {
                std::cerr << "Mismatch at (" << i << ", " << j << "): CPU = " << cpuResult[i][j] << ", CUDA = " << cudaResult[i * n + j] << std::endl;
                return false;
            }
        }
    }
    std::cout << "Results from CPU and CUDA match!\n";
    return true;
}

int main(int argc, char *argv[])
{
    int n = 512;
    double density = 0.3;

    std::cout << "Floyd-Warshall CPU/CUDA Comparison\n";
    std::cout << "------------------------------------\n";
    std::cout << "Arguments: [N (positive integer)] [Density (0.0 to 1.0)]\n";
    std::cout << "------------------------------------\n";
    std::cout << "Limits for N: Positive integer. Recommended up to 2048 for reasonable CPU time.\n";
    std::cout << "Ensure your system has enough RAM for the matrix (N*N*4 bytes).\n";
    std::cout << "Limits for Density: Floating-point number between 0.0 and 1.0.\n";
    std::cout << "------------------------------------\n";

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
        std::cerr << "Warning: Large matrix size. This may lead to excessive CPU execution time and potential memory issues.\n";
    }

    std::cout << "Generating a " << n << "x" << n << " random adjacency matrix with density " << density << "...\n";
    std::vector<std::vector<int>> graph_host = generateRandomMatrix(n, density);
    std::cout << "Matrix generated.\n";

    std::vector<std::vector<int>> graph_cuda_input = graph_host;
    int *d_in = new int[n * n];
    int *d_out = new int[n * n];
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            d_in[i * n + j] = graph_cuda_input[i][j];
        }
    }

    std::cout << "Running Floyd-Warshall on CUDA...\n";
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda, 0);

    floydWarshallCUDA(d_in, n, d_out);

    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);
    float milliseconds_cuda = 0;
    cudaEventElapsedTime(&milliseconds_cuda, start_cuda, stop_cuda);
    std::cout << "CUDA Floyd-Warshall execution time: " << milliseconds_cuda << " ms\n";

    std::cout << "Running Floyd-Warshall on CPU...\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> graph_cpu = graph_host;
    floydWarshallCPU(graph_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    std::cout << "CPU Floyd-Warshall execution time: " << duration_cpu.count() << " ms\n";

    bool resultsMatch = compareResults(graph_cpu, d_out, n);

    delete[] d_in;
    delete[] d_out;
    return 0;
}