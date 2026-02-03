/*
THIS IS SOME SIMPLE DEMO CODE SHOWING YOU HOW TO SETUP A TK FILE AND PROJECTs
*/


// Using thunderkittens!
#include "kittens.cuh"
using namespace kittens;

// OUR GLOBAL TILE DEFINITIONS!
const int rows = 32;
const int cols = 16;

static constexpr int TM = 16;
static constexpr int TN = 16;


// These "globals" are a struct that contain the KERNEL INPUT ARGUMENTS
// THESE ARE THE INPUTS TO THE KERNEL FUNCTION: we just pass in this single struct
struct global_defs {
    gl<float, -1, -1, -1, -1, st_fl<TM, TN>> devA;
    gl<float, -1, -1, -1, -1, st_fl<TM, TN>> devB;
    gl<float, -1, -1, -1, -1, st_fl<TM, TN>> devOut;
};


// MY TK KERNEL
__global__ 
void add_tk(const __grid_constant__ global_defs g) {
    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<TM, TN> (&shared_block_A) = al.allocate<st_fl<TM, TN>>();
    st_fl<TM, TN> (&shared_block_B) = al.allocate<st_fl<TM, TN>>();

    // register tiles
    rt_fl<TM, TN> register_block_A;
    rt_fl<TM, TN> register_block_B;

    // COMPUTATIONS
    //      KEY: PRETEND THAT YOU ARE A WARP!!! AND YOU ARE OPERATING ON A BLOCK OF WORK!!!
    //      ___YOU WILL INDEX INTO blockIdx.x/y___, TK JUST PREVENTS YOU FROM HAVING TO DEAL WITH THREADIDX.x MOST OF THE TIME BC. EVERYTHING IS DONE IN TILES ALREADY
    //      Note that you DO NOT HAVE TO THINK ABOUT THREAD.IDX at all here! You just express what you want the tiles to do, and the work gets done somehow someway!
    for (int i = 0; i <= 1; i++) {
        // HBM -> Shared
        warp::load(shared_block_A, g.devA, {0, 0, i, 0});
        warp::load(shared_block_B, g.devB, {0, 0, i, 0});
        __syncthreads();                                    // you need this bc. the threads in the warp are doing some black magic

        // Shared -> Register
        warp::load(register_block_A, shared_block_A);
        warp::load(register_block_B, shared_block_B);
        __syncthreads();

        // Register computation: A = A + B
        warp::add(register_block_A, register_block_A, register_block_B);
        __syncthreads();

        // Register -> Shared -> HBM
        warp::store(shared_block_A, register_block_A);
        __syncthreads();
        warp::store(g.devOut, shared_block_A, {0, 0, i, 0});      // Store to output, not input!
    }
}



void dispatch_add(float* devA, float* devB, float* devOut) {
    // HELPER TYPE
    // This is the "global layout" for HBM memory tensors.
    //      <DataType, batch, heads, seq_len, head_dim, tile_layout>
    //      -1 dimensions means its a runtime determined value. I.e. it's not fixed at compile time, so its flexible!
    //      tile_layout tells TK HOW the data is organized, i.e. how much are we "tiling" by!
    using global_layout_def = gl<float, -1, -1, -1, -1, st_fl<TM, TN>>;
    using global_args = global_defs;


    // WRAPPING STEP
    // TLDR; WE ARE TRANSLATING OUR DEVICE POINTER TO A GLOBAL LAYOUT
    //      i.e. we are "wrapping" the pointer in a nicer tile layout definition so it's easier to work with
    global_layout_def dev_A{devA, 1, 1, rows, cols};
    global_layout_def dev_B{devB, 1, 1, rows, cols};            // that -1 is nice, no? We can say the tile is of any size!
    global_layout_def dev_Out{devOut, 1, 1, rows, cols};        // Added output

    // KERNEL ARGS STEP
    global_args my_kernel_args{dev_A, dev_B, dev_Out};   // MUST fill these in immediately: cannot fill in later!


    // LAUNCHING THE KERNEL
    //      <<<num_blocks, num_threads_per_block, SMEM_allocated_for_this_block>>>
    unsigned long mem_size = 2 * kittens::size_bytes<st_fl<TM, TN>> + 1024;
    add_tk<<<1,32,mem_size>>>(my_kernel_args);
    cudaDeviceSynchronize();
}

int main() {
    // create input tensors on the host

    const int size = rows * cols;

    float* host_A = new float[size];
    float* host_B = new float[size];

    for (int i = 0; i < size; i++) {     // init the data
        host_A[i] = i;
        host_B[i] = i;
    }

    // create/transfer to device memory
    float* device_A;
    float* device_B;
    float* device_Out;
    cudaMalloc(&device_A, size * sizeof(float));
    cudaMalloc(&device_B, size * sizeof(float));
    cudaMalloc(&device_Out, size * sizeof(float));
    cudaMemcpy(device_A, host_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size * sizeof(float), cudaMemcpyHostToDevice);



    // now we can launch our TK kernel
    dispatch_add(device_A, device_B, device_Out);



    // transfer results back and check
    float* host_result = new float[size];
    cudaMemcpy(host_result, device_Out, size * sizeof(float), cudaMemcpyDeviceToHost);

    bool is_correct = true;
    for (int i = 0; i < size; i++) {
        if (fabs(host_result[i] - (host_A[i] + host_B[i])) > 1e-5) {
            is_correct = false;
            break;
        }
    }

    if (is_correct) {
        printf("PASSED!\n");
    } else {
        printf("FAILED!\n");
    }

    delete[] host_A;
    delete[] host_B;
    delete[] host_result;
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_Out);

    return 0;
}