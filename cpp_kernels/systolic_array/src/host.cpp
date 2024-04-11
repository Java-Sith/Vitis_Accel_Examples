/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/*******************************************************************************

Description:

    This is a matrix multiplication which showcases the "Systolic Array" based
    algorithm design. Systolic array type of implementation is well suited for
    FPGAs. It is a good coding practice to convert base algorithm into Systolic
    Array implementation if it is feasible to do so.

*******************************************************************************/
#include "xcl2.hpp"
#include <vector>
#include <chrono>
#include <sstream>

// Array Size to access
#define DATA_SIZE 1280

// Maximum Array Size
#define MAX_SIZE 1536

// Software implementation of Matrix Multiplication
// The inputs are of the size (DATA_SIZE x DATA_SIZE)
void mat_mul(std::vector<int, aligned_allocator<int> >& in1, // Input Matrix 1
                    std::vector<int, aligned_allocator<int> >& in2, // Input Matrix 2
                    std::vector<int, aligned_allocator<int> >& out  // Output Matrix
                    ) {
    // Perform Matrix multiply Out = In1 x In2
    for (int i = 0; i < DATA_SIZE; i++) {
        for (int j = 0; j < MAX_SIZE; j++) {
            for (int k = 0; k < DATA_SIZE; k++) {
                out[i * DATA_SIZE + j] += in1[i * DATA_SIZE + k] * in2[k * DATA_SIZE + j];
            }
        }
    }
}

// Function to load a tensor from a text file
void load_tensor(std::vector<int, aligned_allocator<int> >& source_in1, int rows, int cols, const char *filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error opening file for reading.\n");
        return;
    }

    std::string line;
    float value;
    for (int i = 0; i < rows; i++) {
        if (!std::getline(file, line)) {
            printf("Error: Not enough rows in file.\n");
            return;
        }
        std::istringstream iss(line);
        for (int j = 0; j < cols; j++) {
            if (!(iss >> value)) {
                printf("Error: Non-numeric value encountered.\n");
                return;
            }
            source_in1[i * cols + j] = static_cast<int>(value);
        }
        // Check for extra values on the line
        if (iss >> value) {
            printf("Error: Too many values on a line.\n");
            return;
        }
    }
    // Check for extra lines in the file
    if (std::getline(file, line)) {
        printf("Error: Too many rows in file.\n");
        return;
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    // Allocate Memory in Host Memory
    if (DATA_SIZE > MAX_SIZE) {
        std::cout << "Size is bigger than internal buffer size, please use a "
                     "size smaller than "
                  << MAX_SIZE << "!" << std::endl;
        return EXIT_FAILURE;
    }

    size_t matrix_size1 = DATA_SIZE * DATA_SIZE;
    size_t matrix_size2 = DATA_SIZE * MAX_SIZE;
    size_t matrix_size_bytes1 = sizeof(int) * matrix_size1;
    size_t matrix_size_bytes2 = sizeof(int) * matrix_size2;
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_systolic_array;

    std::vector<int, aligned_allocator<int> > source_in1(matrix_size1);
    std::vector<int, aligned_allocator<int> > source_in2(matrix_size2);
    std::vector<int, aligned_allocator<int> > source_hw_results(matrix_size2);
    std::vector<int, aligned_allocator<int> > source_sw_results(matrix_size2);
    // Create a new vector to hold the data read from the buffer
    std::vector<int, aligned_allocator<int> > read_data(matrix_size1);

    load_tensor(source_in1, DATA_SIZE, DATA_SIZE, "tensor1.txt");
    load_tensor(source_in2, DATA_SIZE, MAX_SIZE, "tensor2.txt");

    // Create the software result
    for (size_t i = 0; i < matrix_size2; i++) {
        source_sw_results[i] = 0;
        source_hw_results[i] = 0;
    }

    // Print the first few values of the matrix to verify that they were loaded correctly
    for (int i = 0; i < std::min(10, DATA_SIZE); i++) {
        for (int j = 0; j < std::min(10, DATA_SIZE); j++) {
            printf("%d ", source_in1[i * DATA_SIZE + j]);
        }
        printf("\n");
    }

    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_systolic_array = cl::Kernel(program, "mmult", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, matrix_size_bytes1,
                                         source_in1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, matrix_size_bytes2,
                                         source_in2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, matrix_size_bytes2,
                                            source_hw_results.data(), &err));

    int a_row = DATA_SIZE;
    int a_col = DATA_SIZE;
    int b_col = MAX_SIZE;

    OCL_CHECK(err, err = krnl_systolic_array.setArg(0, buffer_in1));
    OCL_CHECK(err, err = krnl_systolic_array.setArg(1, buffer_in2));
    OCL_CHECK(err, err = krnl_systolic_array.setArg(2, buffer_output));
    OCL_CHECK(err, err = krnl_systolic_array.setArg(3, a_row));
    OCL_CHECK(err, err = krnl_systolic_array.setArg(4, a_col));
    OCL_CHECK(err, err = krnl_systolic_array.setArg(5, b_col));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));

    // Read the data from the buffer
    OCL_CHECK(err, q.enqueueReadBuffer(buffer_in1, CL_TRUE, 0, matrix_size_bytes1, read_data.data(), nullptr, nullptr));
    q.finish();

    // Compare the read data with the original data
    for (size_t i = 0; i < matrix_size1; i++) {
        if (source_in1[i] != read_data[i]) {
            printf("Error: Data mismatch at index %d. Expected %d, got %d.\n", i, source_in1[i], read_data[i]);
        }
    }

    printf("Data verification successful. No mismatches found.\n");

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_systolic_array));
    q.finish();

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OPENCL HOST CODE AREA END

    // Compute Software Results
    mat_mul(source_in1, source_in2, source_sw_results);

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();

    //Calculate time
    std::chrono::duration<double> duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::cout << "Execution time: " << duration_ms.count() << " milliseconds" << std::endl;


    // Compare the results of the Device to the simulation
    int match = 0;
    for (size_t i = 0; i < matrix_size2; i++) {
        if (source_hw_results[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                      << " Device result = " << source_hw_results[i] << std::endl;
            match = 1;
            break;
        }
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}
