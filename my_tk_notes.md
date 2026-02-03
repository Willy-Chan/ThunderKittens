# TK

Provides a bunch of *TEMPLATED WRAPPERS* over CUDA: so it feels more pytorchlike.

# TLDR;
When you are in a TK kernel, **pretend you are a warp/warpgroup** that's operating on a block of data. You load in _TILES_ from global memory, and your warps operate on those <16 x 16> (or multiple of those dimensions) tiles without you having to deal with threadIdx yourself. **KEY POINT: YOU DON'T HAVE TO DEAL WITH THREADIDX FOR OPERATING ON DATA!**

Each kernel is thus a "warp" that is operating on a **block** of work. So given your block of work, start tiling the global memory, and then have your warps do stuff on that global memory! No thread indexing needed (unless you want to do more complicated LCSF stuff).

OH by the way:
Blocks can have multiple warps, i.e.
``` cpp
add_tk<<<4, 128>>>(args);  // 4 blocks × 4 warps per block
```
This way you can have multiple warps in the same block "do" stuff in parallel: up to 4 warps can do stuff simultaneously on an H100!

```cpp
// Each warp can work on different data:
int warp_id = threadIdx.x / 32;  // 0, 1, 2, or 3
int tile_idx = blockIdx.x * 4 + warp_id;  // Global tile index

// Warp 0: tile 0, Warp 1: tile 1, Warp 2: tile 2, Warp 3: tile 3
warp::load(tile, g.input, {0, 0, tile_idx, 0});     // All 4 warps load DIFFERENT tiles in parallel
```
Each warp has its own registers, they do NOT share them. But they share the same shared memory pool. 



# Full Simplest Example
```cpp
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
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
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
#include "harness.impl"

```


## Tiles
```cpp
// Tile of data that lives in shared memory

// TLDR; THIS IS JUST SOMEBOILERPLATE TO ENSURE OUR MEMORY IS ALIGNED (via alignment_dummy)
extern __shared__ alignment_dummy __shm[];      // __shm[] is a shared memory array whose size will be determined at kernel launch time
shared_allocator al((int*)&__shm[0]);       // "al" is some object that will be managing our shared memory. This is how you actually allocate it. 

#define _row 16
#define _col 32

// each tile goes: <NUM_ROWS, NUM_COLS>. Also has a datatype (st_float) in this cases. Defining x_s and o_s as our shared memory tiles.
// We allocate float tiles via al.allocate
st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

// Register Tile: of a certain size.
rt_fl<_row, _col> x_reg_fl;
```

## Loading and Storing from HBM
```cpp
// load from global to shared
// (batch, head, seq, dimension) notation
load(x_s, g.x, {0, 0, 0, 0});  // Specify the STARTING COORDINATE. x_s will get filled with however much data it needs (16x32)
 __syncthreads();

// Go to position batch=0, head=0, s=0, d=0
// Take 16 elems along seq dimension, 32 elems along dim dimension
// Copy that slice into x_s
// MEMORY IS ROW-MAOR BUT YOU THINK IN TILES, COORDS TELL YOU WHERE TILES START

load(x_s, g.x, {0, 0, 16, 0}); // tile BELOW (next 16 rows, same cols)
 __syncthreads();

load(x_s, g.x, {0, 0, 0, 32}); // tile RIGHT (next 32 cols, same rows)
 __syncthreads();
```

Can also load from shared to register:
```cpp
load(x_reg_fl, x_s);   // ASSUMES SAME DIMENSIONS!
__syncthreads();
```

Storing is a very similar story:
```cpp
// x (dst) = x (src b) + x (src a)
add(x_reg_fl, x_reg_fl, x_reg_fl);
__syncthreads();

// store from register to shared
store(o_s, x_reg_fl);
__syncthreads();

// store from shared to HBM
store(g.o, o_s, {0, 0, 0, 0});  // (0,0,0,0) is the destination coordinate where my tile gets stored
__syncthreads();
```

# LAUNCHING TK KERNELS
