//
//  MetalKernelDispatcher.swift
//  AppleMetalDemo
//
//  Created by Manish Kumar on 2025-07-03.
//

import UIKit
import Metal

class MetalKernelDispatcher {

    /// Execute a simple GPU compute operation using Metal to add two arrays of floats.
    /// This function demonstrates the full end-to-end workflow of compute programming with Metal - from buffer creation and
    /// parameter encoding to kernel dispatch and result extraction.
    func executeMetalKernel() -> (String, [Float]) {
        do {
            let metalKernelHelper = try MetalKernelHelper(functionName: "vector_add")
            let count = 1024_000
            let A = Array(0..<count).map { Float($0) }
            let B = Array(0..<count).map { Float($0 * 10) }

            // A buffer is just an area of memory on the GPU. Here we are reserving storage on the device global memory for the input arrays
            // and then copying data into them from the host (CPU) side.
            let bufferA = metalKernelHelper.makeBuffer(from: A)
            let bufferB = metalKernelHelper.makeBuffer(from: B)
            let bufferC = metalKernelHelper.emptyBuffer(count: count, of: Float.self)

            // This struct will also exist on the Metal side.
            struct Params {
                var count: UInt32
            }

            // Create a buffer on the device to hold this struct and copy the value of `count` from host to this buffer.
            // Note: It's not necessary to create a struct to pass scalar/constant values to the GPU but it's recommended to do so especially
            // when there are multiple values to be passed. Otherwise we'd need to create a separate buffer for each constant and each constant
            // will take up one buffer slot as well.
            let constants = metalKernelHelper.makeConstant(from: Params(count: UInt32(count)))

            // Note down the start time before dispatch
            let start = CACurrentMediaTime()

            // Invoke the Metal kernel to do the actual data processing (array addition in this case).
            // Notice the use of `dispatchThreadgroups`. Another method to invoke a kernel is `dispatchThreads`
            // but the use of `dispatchThreadgroups` is preferrable as it allows more control over thread indexing
            // and the use of shared memory for optimizations.
            metalKernelHelper.dispatchThreadgroups(buffers: [bufferA, bufferB, bufferC],
                                                   constants: [constants],
                                                   count: count)

            // Note down the end time after dispatch
            let end = CACurrentMediaTime()
            // Print the time in milliseconds
            let elapsedMs = (end - start) * 1000
            let kernelExecutionTimeString = String(format: "%.3f ms", elapsedMs)
            print(String(format: "Kernel execution time: %.3f ms", elapsedMs))

            // Get a pointer to the contents of the output GPU buffer.
            // The GPU buffer is raw memory as it does not have any type etc. It's up to us to interpret it based on the
            // problem at hand. We use `bindMemory` in Swift for this purpose and interpret it as an array
            // of `Float`s in this case.
            let outputPtr = bufferC.contents().bindMemory(to: Float.self, capacity: count)

            // Convert the pointer contents into a regular Swift array using an `UnsafeBufferPointer`.
            let result = Array(UnsafeBufferPointer(start: outputPtr, count: count))
            return (kernelExecutionTimeString, result)
        } catch let error {
            print("Metal error: \(error)")
            return ("Error", [])
        }
    }
}
