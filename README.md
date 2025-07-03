# Apple Metal - Vector Addition Demo

This project is the "Hello, World" of GPU compute programming. It demonstrates how to add two arrays of length N by taking advantage of GPU parallelism, where each of the N threads is responsible for adding a pair of elements.

This repository is a minimal, educational example that demonstrates how to use Apple's Metal framework for GPU programming in Swift. It's designed to help you understand the full end-to-end pipeline of compute programming with Metal in a real Swift app. The code includes **extensive inline comments** explaining the functioning of each step and design choices made.

## Topics Covered

The projet shows how to:

- Set up a basic Metal compute pipeline in Swift
- Write and compile a Metal kernel (`.metal` file)
- Transfer data between CPU and GPU using Metal buffers
- Dispatch GPU workloads with `dispatchThreadGroups`
- Get results back from the GPU

### License

MIT License — free to learn from and build on.


### Author

Created by Manish Kumar

Questions welcome!

