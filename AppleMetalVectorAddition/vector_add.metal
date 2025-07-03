#include <metal_stdlib>
using namespace metal;

// Note: Metal does not carry buffer length information inside the GPU function. The device buffer is just a
// raw memory address from the GPU’s perspective — it has no metadata about size, bounds, or structure. That's why we
// cannot determine the length of Array A here from the buffer itself and have to pass the count using `Params`.
struct Params {
    uint count;
};

// Metal kernel to add two input arrays element-wise and store the result in a third array.
// Note: The order of buffers A, B, and C in the parameter list must exactly match how they are bound
// on the Swift side using setBuffer(..., offset: 0, index: N).
kernel void vector_add(device const float* A [[ buffer(0) ]],
                       device const float* B [[ buffer(1) ]],
                       device float* C       [[ buffer(2) ]],
                       constant Params& params [[ buffer(3) ]],
                       uint3 threadIdx [[ thread_position_in_threadgroup ]], // Same variable naming convention as CUDA
                       uint3 blockIdx  [[ threadgroup_position_in_grid ]],   // Same variable naming convention as CUDA
                       uint3 blockDim  [[ threads_per_threadgroup ]])        // Same variable naming convention as CUDA
{
    // Compute the global thread index within the entire grid. This method of computing the gload thread index manually gives us
    // more control over thread indexing and we can optimize it based on the nature of the problem. Depending on built-in variable
    // `thread_position_in_grid` can sometimes lead to unexpected results as the way Metal runtime flattens the multi-dimensional
    // input data may not be same as how we'd like it to be interpreted.
    uint globalIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Doing the bounds check here is important as `dispatchThreadgroups` might launch more threads than are required to cover
    // the input data if the total number of threads (array length) is not an exact multiple of the threadgroup size. Performing
    // this check ensures that any extra threads do not participate in the computation.
    if (globalIdx < params.count) {
        C[globalIdx] = A[globalIdx] + B[globalIdx];
    }
}
