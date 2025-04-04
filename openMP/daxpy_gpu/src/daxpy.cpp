#include <iostream>
#include <omp.h>

void daxpy(int n, double a, double* x, double* y) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int n_experiments = 10;
    const int n = 1000000000;
    double a = 2.0;
    double *x = new double[n];
    double *y = new double[n];

    // Initialize arrays
    for (int i = 0; i < n; ++i) {
        x[i] = i * 0.5;
        y[i] = i * 0.3;
    }

       // allocate the device memory
   #pragma omp target data map(to:x[0:n]) map(tofrom:y[0:n])
   {

        for (int i=0; i<n_experiments; i++){
            // Perform DAXPY computation
            daxpy(n, a, x, y);
        }

   }

    // Print some results
    std::cout << "y[0] = " << y[0] << std::endl;
    std::cout << "y[n-1] = " << y[n-1] << std::endl;

    delete[] x;
    delete[] y;

    return 0;
}
