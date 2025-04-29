Description:The Floyd-Warshall algorithm is a fundamental graph algorithm for finding the shortest paths between all pairs of vertices in a weighted graph. This classic dynamic programming approach has a time complexity of O(V³), making it computationally intensive for large graphs. This project explores parallel implementations of the Floyd-Warshall algorithm to address these performance limitations.
We implemented and compared three versions of the algorithm:

1. A sequential CPU implementation (baseline)
2. A parallel implementation using OpenMP for CPU multithreading
3. A parallel implementation using CUDA for GPU acceleration
   By parallelizing the algorithm on both multicore CPUs and GPUs, we aim to demonstrate significant performance improvements while maintaining computational correctness. The project evaluates each implementation's execution time across various graph sizes and densities, highlighting the scalability and efficiency benefits of parallel computing approaches.

#### Repository: https://github.com/MakarandPundlik/CSC548

#### CUDA

Compilation: nvcc floyd-warshall.cu -o floyd-warshall -lcuda -std=c++11
Execution: ./floyd-warshall <number of vertices> <density>
Dependencies: CUDA Toolkit: NVIDIA's CUDA Toolkit must be installed on your system. This includes the CUDA compiler (nvcc), runtime libraries, and headers.
A CUDA-capable GPU: You'll need a GPU that supports CUDA to run the CUDA implementation.

#### OpenMP

Compilation: g++ -fopenmp floyd-warshall-openmp.cpp -o floyd-warshall-openmp
Execution: ./floyd-warshall-openmp <number of vertices> <density>
Dependencies: A C++ compiler with OpenMP support: You'll need a compiler like GCC or Clang that supports OpenMP.
OpenMP runtime libraries: These are usually included with the compiler.

#### Team Members: Makarand Pundlik
