__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" void add_kernel(float *a, float *b, float *c, int n) {

    // float *a_d;
    // float *b_d;
    // float *c_d;
    // CUDA_CHECK(cudaMalloc(&a_d, n * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&b_d, n * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&c_d, n * sizeof(float)));
    // CUDA_CHECK(cudaMemset(c_d, 0, n * sizeof(float)));
    // CUDA_CHECK(cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice));

    // add<<<blocks, threads>>>(a_d, b_d, c_d, n);
    // CUDA_CHECK(cudaDeviceSynchronize());  // wait for kernel to finish

    // CUDA_CHECK(cudaMemcpy(c, c_d, n * sizeof(float), cudaMemcpyDeviceToHost));
    // cudaFree(a_d);
    // cudaFree(b_d);
    // cudaFree(c_d);

    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    // }
}