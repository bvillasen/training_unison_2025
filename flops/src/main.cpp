#include <stdio.h>
#include <iostream>
#include <hip/hip_runtime.h>

template<int n_iter>
__global__ void flops_add(double *input, const int n) {
    const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t n_threads  = gridDim.x * blockDim.x;
    const int n_values_per_thread = n / n_threads;
    const uint64_t max_offset = n_values_per_thread * n_threads;

    double *ptr;
    const double y = 1.0;

    ptr = &input[gid];
    double x = 2.0;

    // For every vector element, its doing one read
    // For every vector element, its doing 2 * iter flops
    // For every thread, its doing one write
    
    for (uint64_t offset = 0; offset < max_offset; offset += n_threads) {
        for (int i = 0; i < n_iter; i++) {
            x = ptr[offset] * x + y;
        }
    }
    ptr[0] = -x;
}

int main(int argc, char* argv[]) {
  
  int n = 134217728;
  int n_experiments = 1000;
  if (argc > 1) n_experiments = std::stoll(argv[1]);
  
  std::cout << "Vector length: " << n << std::endl;
  std::cout << "N experiments: " << n_experiments << std::endl;

  int block_size = 256;
  int grid_size = 228 * 128; 
  int n_threads = grid_size * block_size;
  uint64_t n_iter = (int) N_ITER;
  std::cout << "Grid size: " << grid_size << std::endl;
  std::cout << "Block size: " << block_size << std::endl;
  std::cout << "Number of threads: " << n_threads << std::endl;
  std::cout << "Number of elements per thread: " << n / (grid_size * block_size) << std::endl;
  std::cout << "Number of iterations: " << n_iter << std::endl;
  
  
  
  double *h_data = (double *) malloc(n*sizeof(double));
  for (int i=0; i<n; i++ ) h_data[i] = (double) i;
  
  double *d_data;
  hipMalloc((void **)&d_data, n*sizeof(double));  
  
  // Events for timing
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  
  // Warmup kernel launch
  flops_add<N_ITER><<<grid_size, block_size>>>( d_data, n );
  
  hipEventRecord(start);
  for (int i=0; i<n_experiments; i++){
    flops_add<N_ITER><<<grid_size, block_size>>>( d_data, n );  
  }  
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float elapsed_time;
  hipEventElapsedTime(&elapsed_time, start, stop);
  
  uint64_t n_flops = 2 * n * n_iter;
  uint64_t n_bytes_moved = 8 * (n + n_threads);
  double arithmetic_intensity = static_cast<double>(n_flops) / n_bytes_moved;
  double flops_rate = n_flops / (elapsed_time/n_experiments);
  double bandwidth = n_bytes_moved / (elapsed_time/n_experiments);
  
  std::cout << "Number of FP64 Flops: " << n_flops << std::endl;
  std::cout << "Number of bytes moved: " << n_bytes_moved << std::endl << std::endl;
  std::cout << "Arithmetic Intensity: " << arithmetic_intensity << std::endl;
  std::cout << "N FP64 Flops/sec: " << flops_rate / 1e9 << " GFlops/s" << std::endl;
  std::cout << "Bandwidth: " << bandwidth / 1e9 << " GB/s" << std::endl;
  
  
  free(h_data);
  hipFree(d_data);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  
  std::cout << "Finished" << std::endl;
  
}