/*
THIS IS SOME SIMPLE DEMO CODE SHOWING YOU HOW TO SETUP A TK FILE AND PROJECTs
*/


// Using thunderkittens!
#include "kittens.cuh"
using namespace kittens;

// OUR GLOBAL TILE DEFINITIONS!
int rows = 32;
int cols = 16;


// These "globals" are a struct that contain the KERNEL INPUT ARGUMENTS
// THESE ARE THE INPUTS TO THE KERNEL FUNCTION: we just pass in this single struct
struct micro_globals {
    gl<float, -1, -1, -1, -1, st_fl<_row, _col>> devA; 
    gl<float, -1, -1, -1, -1, st_fl<_row, _col>> devB;
};


void dispatch_add(float* devA, float* devB) {

    // This is the "global layout" for HBM memory tensors.
    //      <DataType, batch, heads, seq_len, head_dim, tile_layout>
    //      -1 dimensions means its a runtime determined value. I.e. it's not fixed at compile time, so its flexible!
    //      tile_layout tells TK HOW the data is organized, i.e. how much are we "tiling" by!
    using global_layout_def = gl<float, -1, -1, -1, -1, st_fl<rows, cols>>;
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
    cudaMalloc(&device_A, size);
    cudaMalloc(&device_B, size);
    cudaMemcpy(device_A, host_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size * sizeof(float), cudaMemcpyHostToDevice);



    // now we can launch our TK kernel
    dispatch_add(device_A, device_B);



    // transfer results back and check
    float* host_A_tocheck = new float[size];
    float* host_B_tocheck = new float[size];
    cudaMemcpy(host_A_tocheck, device_A, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B_tocheck, device_B, size * sizeof(float), cudaMemcpyDeviceToHost);

    bool is_correct = true;
    for (int i = 0; i < size; i++) {
        if (host_A_tocheck[i] != host_A[i] || host_B_tocheck[i] != host_B[i]) {
            is_correct = false;
        }
    }

    if (is_correct) {
        printf("PASSED!\n");
    } else {
        printf("FAILED!\n");
    }

    delete[] host_A;
    delete[] host_B;
    delete[] host_A_tocheck;
    delete[] host_B_tocheck;
    cudaFree(device_A);
    cudaFree(device_B);

    return 0;
}