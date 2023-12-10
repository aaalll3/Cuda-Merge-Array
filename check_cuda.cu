#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0); // Attempt to set the device to the first CUDA device (change the index as needed)

    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Check the number of CUDA devices
    int deviceCount;
    cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // Print information about each CUDA device
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaStatus = cudaGetDeviceProperties(&deviceProp, deviceId);

        if (cudaStatus != cudaSuccess) {
            std::cerr << "Error getting CUDA device properties: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }

        std::cout << "CUDA Device " << deviceId << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        // Add more information as needed
        std::cout << std::endl;
    }

    return 0;
}
