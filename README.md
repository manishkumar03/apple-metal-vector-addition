# Apple Metal - Vector Addition Demo

This project is the "Hello, World" of GPU compute programming. It demonstrates how to add two arrays of length N by taking advantage of GPU parallelism, where each of the N threads is responsible for adding a pair of elements.

This repository is a minimal, educational example that demonstrates how to use Apple's Metal framework for GPU programming in Swift. It's designed to help you understand the full end-to-end pipeline of compute programming with Metal in a real Swift app. It shows how to prepare data on the CPU, send it to the GPU, run a Metal kernel, and read back the results.

The code includes **extensive inline comments** explaining the functioning of each step and the design choices made.

The project includes two core classes:

1. 	`MetalKernelHelper.swift`: All the boilerplate for setting up Metal runtime. 

	It handles:
 	* 	Metal device creation
 	* 	Kernel loading from the default Metal library
 	* 	Compute pipeline setup
 	* 	Command buffer + encoder setup
 	* 	Utility methods for creating buffers from arrays or constants

 	
2. `MetalKernelDispatcher.swift`: For dispatching the `vector_add` kernel to the GPU for execution.

	It handles:
 	* 	Input array preparation (two [Float] arrays)
 	* 	Data loading into GPU buffers
 	*  Invoking the Metal kernel for execution
 	* 	Timing the kernel execution
 	* 	Reading back the results into a Swift array


## Topics Covered

The projet shows how to:

- Set up a complete Metal compute pipeline in Swift
- Write and compile a Metal kernel (`.metal` file)
- Transfer data between CPU and GPU using Metal buffers
- Dispatch GPU workloads with `dispatchThreadGroups`
- Get results back from the GPU

### License

MIT License — free to learn from and build on. A star ⭐️ would be awesome if this helped you!


### Author

Created by Manish Kumar

Questions welcome!

