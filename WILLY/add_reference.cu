#include "kittens.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

struct micro_globals {
    using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    _gl x, o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;

    // load from HBM to shared
    warp::load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    warp::load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    warp::add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    warp::store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    warp::store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}

void dispatch_micro( float *d_x, float *d_o ) {
    using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    using globals = micro_globals;
    _gl  x_arg{d_x, 1, 1, _row, _col};
    _gl  o_arg{d_o, 1, 1, _row, _col};
    globals g{x_arg, o_arg};
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<1,32,mem_size>>>(g);
    cudaDeviceSynchronize();
}

int main() {
    const int size = _row * _col;
    float *h_x = new float[size];
    float *h_o = new float[size];
    float *d_x, *d_o;

    // Initialize input data
    for(int i = 0; i < size; i++) {
        h_x[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_o, size * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice);

    // Run the kernel
    dispatch_micro(d_x, d_o);

    // Copy result back to host
    cudaMemcpy(h_o, d_o, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for(int i = 0; i < size; i++) {
        if(fabs(h_o[i] - (2.0f * h_x[i])) > 1e-5) {
            correct = false;
            break;
        }
    }

    if(correct) {
        printf("Test passed!\n");
    } else {
        printf("Test failed!\n");
    }

    // Free memory
    delete[] h_x;
    delete[] h_o;
    cudaFree(d_x);
    cudaFree(d_o);

    return 0;
}

