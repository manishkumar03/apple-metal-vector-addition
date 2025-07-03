//
//  MetalKernelHelper.swift
//  AppleMetalDemoApp
//
//  Created by Manish Kumar on 2025-06-27.
//

import Metal

class MetalKernelHelper {
    let device: MTLDevice
    let pipeline: MTLComputePipelineState
    let commandQueue: MTLCommandQueue

    init(functionName: String) throws {
        // Get a handle to the GPU. There is only one GPU on iPhones and Apple Silicon Macs but
        // there might be more than one on older Intel-based Macs.
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalErrors.unsupportedDevice
        }

        // A Metal library is a compiled collection of `*.metal` kernels which are embedded into the
        // app bundle by Xcode during the build process. This function can return nil if the kernels did not compile correctly.
        guard let library = try? device.makeDefaultLibrary(bundle: .main) else {
            throw MetalErrors.libraryCreationFailed
        }

        // Try to extract a specific Metal kernel function (by name) from the already loaded `MTLLibrary`.
        // It can return nil if the name doesn't match any function in the `*.metal` files.
        guard let kernelfunction = library.makeFunction(name: functionName) else {
            throw MetalErrors.makeFunctionFailed
        }


        // Try to create a `MTLCommandQueue` from a `MTLDevice`. A command queue is a GPU instruction scheduler.
        // It's how you submit work (compute commands) to the GPU. See the function `dispatchThreadgroups()` for details.
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalErrors.commandQueueCreationFailed
        }

        self.device = device
        self.commandQueue = commandQueue

        // Compile the `kernelFuction` into a compute pipeline state of type `MTLComputePipelineState` that the
        // GPU can execute. Basically, `makeComputePipelineState()` tells Metal to convert this kernel function
        // into something which the GPU can execute efficiently.
        self.pipeline = try device.makeComputePipelineState(function: kernelfunction)
    }

    /// Dispatches a compute kernel using the precompiled pipeline state, binding the provided buffers and constants,
    /// and launching GPU threads with specified configuration.
    ///
    /// - Parameters:
    ///   - buffers: Input/output `MTLBuffer`s to be bound to the kernel.
    ///   - constants: Optional constant `MTLBuffer`s, such as uniform or metadata inputs (default is empty).
    ///   - count: Total number of elements or threads to dispatch.
    ///   - threadsPerGroup: Number of threads per threadgroup (default is 32).
    func dispatchThreadgroups(buffers: [MTLBuffer],
                              constants: [MTLBuffer] = [],
                              count: Int,
                              threadsPerGroup: Int = 32) {

        // Create a command buffer and compute command encoder from the command queue. These are needed to
        // encode and submit work to the GPU. Each command buffer contains a sequence of instructions which are
        // enriched by the encoder with other required information, including:
        //        - Which kernel function to run
        //        - Bind buffers, textures, and other resources
        //        - Specify how many threads to dispatch (with dispatchThreadgroups)
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or encoder")
        }

        // Set the compute pipeline state that contains the compiled kernel function. i.e. It tells the compute encoder
        // which compiled Metal kernel (pipeline) to execute when dispatching threads. This step is required before calling
        // dispatchThreadgroups(...), or the GPU won’t know which function to execute.
        encoder.setComputePipelineState(self.pipeline)

        // A buffer is a contiguous block of memory on the GPU which is used to store data such as input arrays, output results,
        // constants, and image data etc.
        //
        // Bind each input/output buffer to its corresponding buffer index. Here `buffers` contains `[bufferA, bufferB, bufferC]`.
        // The order of buffers set here should match the buffers defined in the kernel.
        for (i, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: i)
        }

        // Bind any additional constant buffers immediately after the main buffers.
        // These may contain parameters or configuration data. The order of constants set here should match the
        // constants defined in the kernel. Note the index value which starts after buffers which means the constants
        // should be defined in the kernel after the buffers.
        for (j, constant) in constants.enumerated() {
            encoder.setBuffer(constant, offset: 0, index: buffers.count + j)
        }

        // Define how many threads each threadgroup will contain (1D layout). Here is set to 32 which is common warp size
        // on GPUs, thus makeing the execution highly performant.
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)

        // Compute the number of threadgroups needed to cover the total thread count.
        // Uses ceiling division to ensure full coverage. Since the total number of threads to be launched may not be an exact
        // multiple of the threadgroup size, this calculation might allocate more blocks than are strictly required
        // for the number of threads that we actually want to launch. We verify each thread's index in the kernel to avoid
        // processing data which is out of bounds.
        let threadgroups = MTLSize(width: (count + threadsPerGroup - 1) / threadsPerGroup,
                                   height: 1, depth: 1)

        // Dispatch the threadgroups to the GPU using the specified configuration.
        // Note the use of `dispatchThreadgroups` here. Other option is to use `dispatchThreads`. We use `dispatchThreadgroups`
        // as it allows advanced optimization techniques to be used based on shared memory and memory coalescing etc. The only
        // downside to this method is that we have to calculate the thread index manually inside the kernel which is a small price
        // to pay for the optimization techniques that it allows us to use.
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        // Finalize end encoding and commit the command buffer to the GPU for execution.
        encoder.endEncoding()
        commandBuffer.commit()

        // Block the CPU in this program until GPU work is finished (usually only needed for synchronous execution). It's ok to block the CPU
        // here as we want to wait for the results so that we can display them on the screen. We typically won't do this in production.
        //
        // We can also use `commandBuffer.addCompletedHandler { buffer in ... }` to do something asynchronously when the GPU finishes. This
        // version is non-blocking and is a good place to check results.
        commandBuffer.waitUntilCompleted()
    }

    /// Creates a Metal buffer from a Swift array of any type `T`.
    /// The buffer is created on the current Metal device and contains a copy of the array's data.
    /// This is a generic utility function used to pass data to the GPU from CPU.
    func makeBuffer<T>(from array: [T]) -> MTLBuffer {
        // Use Swift's withUnsafeBytes to access the raw memory of the array.
        return array.withUnsafeBytes { rawBuffer in
            // Create a Metal buffer using the raw memory pointer and the byte count.
            // options: [] means no special storage or caching options are applied.
            // The '!' assumes buffer creation succeeds — should be safe if device is valid and memory is sufficient.
            device.makeBuffer(bytes: rawBuffer.baseAddress!,
                              length: rawBuffer.count,
                              options: [])!
        }
    }

    // Creates an empty Metal buffer capable of holding `count` elements of type `T`.
    // Useful for allocating output or intermediate buffers on the GPU.
    func emptyBuffer<T>(count: Int, of type: T.Type = T.self) -> MTLBuffer {
        // Compute the total buffer size in bytes and allocate a Metal buffer accordingly.
        // We use `stride` (not `size`) to ensure that each element of type `T` is properly aligned in memory,
        // including any padding required for alignment on word boundaries.
        device.makeBuffer(length: count * MemoryLayout<T>.stride, options: [])!
    }

    // Creates a Metal buffer containing a single constant value of type `T`.
    // This is typically used to pass small uniform values to a Metal kernel.
    func makeConstant<T>(from value: T) -> MTLBuffer {
        var copy = value
        // Use withUnsafeBytes to get a raw pointer to the constant value.
        return withUnsafeBytes(of: &copy) { rawBuffer in
            // Create a Metal buffer that holds the value.
            // The contents are copied from the raw buffer into GPU memory.
            device.makeBuffer(bytes: rawBuffer.baseAddress!,
                              length: rawBuffer.count,
                              options: [])!
        }
    }}
