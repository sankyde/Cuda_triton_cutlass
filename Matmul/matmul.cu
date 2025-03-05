#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Macro for CUDA error checking. It prints the error message and exits on error.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void simpleMatMul(float *A,float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N){
        float sum = 0.0f;
        for(int k=0;k<N;++k){
            sum += A[row*N + k] * B[k*N + col];
        }
        C[row*N+col] = sum;
    }
}

int main(){
    int N = 512;
    size_t bytes = N * N * sizeof(float); 

    float *h_A, *h_B, *h_C;
    CUDA_CHECK( cudaMallocHost((void**)&h_A,bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_B,bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_C,bytes));

    for(int i=0;i<N*N;++i){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B,*d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A,bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B,bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C,bytes));

    CUDA_CHECK(cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice));


    //configure the block and grid dim
    int tile_dim=16;
    dim3 blockDim(tile_dim,tile_dim);
    dim3 gridDim((N+tile_dim-1)/tile_dim,(N+tile_dim - 1)/tile_dim);


    //Create cuda events for timing the kernel
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    //the cuda event record, records the event 
    CUDA_CHECK(cudaEventRecord(startEvent,0));

    simpleMatMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stopEvent,0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    //Calculate time
    float elapsedTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime,startEvent,stopEvent));
    printf("Kernel Execution Time %3f ms\n",elapsedTime);

    CUDA_CHECK(cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost));

    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[%d][%d] = %f\n", N - 1, N - 1, h_C[(N - 1) * N + (N - 1)]);

    // Clean up: free device and host memory, and destroy events
    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );
    CUDA_CHECK( cudaFreeHost(h_A) );
    CUDA_CHECK( cudaFreeHost(h_B) );
    CUDA_CHECK( cudaFreeHost(h_C) );
    CUDA_CHECK( cudaEventDestroy(startEvent) );
    CUDA_CHECK( cudaEventDestroy(stopEvent) );

    return 0;
    
}