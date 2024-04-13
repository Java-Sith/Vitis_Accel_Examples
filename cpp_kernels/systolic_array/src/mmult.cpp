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

Vitis Key Concept :

    This is a matrix multiplication example which showcases the "Systolic Array"
    based algorithm design. Systolic array type of implementation is well suited
    for FPGAs.

*******************************************************************************/

/*

Kernel Description :

    This kernel is a systolic array based matrix multiplication. Though the
    maximum size of the input matrices are restricted to a smaller MAX_SIZE, it
    is still possible to use this approach and get better performance for larger
    matrices by using tiling.

    Arguments :

        int *a     (input )  --> Input  Matrix A
        int *b     (input )  --> Input  Matrix B
        int *c     (output)  --> Output Matrix
        int  a_row (input )  --> Row Size Matrix A
        int  a_col (input )  --> Col Size Matrix A
        int  b_col (input )  --> Col Size Matrix B

    Kernel Configuration :

        Max Size    --> 16

    Note :
        Max Size is dependent on the available DSP resources in the FPGA
*/

#include <stdio.h>

// Maximum Array Size
#define MAX_SIZE 1536
#define DATA_SIZE 1280

// TRIPCOUNT identifier
#define TILE_SIZE 16

extern "C" {
void mmult(const int* a, // Read-Only Matrix A
           const int* b, // Read-Only Matrix B
           int* c,       // Output Result
           int a_row,    // Matrix A Row Size
           int a_col,    // Matrix A Col Size
           int b_col     // Matrix B Col Size
           ) {
    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    // Local memory to store input and output matrices
    int localA[DATA_SIZE][DATA_SIZE];
//#pragma HLS ARRAY_PARTITION variable = localA dim = 1 complete

    int localB[DATA_SIZE][MAX_SIZE];
//#pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete

    int localC[DATA_SIZE][MAX_SIZE];
//#pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete

// Burst reads on input matrices from global memory
// Read Input A
// Auto-pipeline is going to apply pipeline to these loops
readA:
    for (int ii = 0; ii < a_row; ii += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min = (DATA_SIZE * DATA_SIZE)/TILE_SIZE max = (MAX_SIZE * MAX_SIZE)/TILE_SIZE
        for (int jj = 0; jj < a_col; jj += TILE_SIZE) {
            // Procesa cada tile
            for (int i = ii; i < min(ii + TILE_SIZE, a_row); i++) {
                for (int j = jj; j < min(jj + TILE_SIZE, a_col); j++) {
                    localA[i][j] = a[i * a_col + j];
                }
            }
        }
    }

// Read Input B
readB:
    for (int ii = 0; ii < b_row; ii += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min = (DATA_SIZE * DATA_SIZE)/TILE_SIZE max = (MAX_SIZE * MAX_SIZE)/TILE_SIZE
        for (int jj = 0; jj < b_col; jj += TILE_SIZE) {
            // Procesa cada tile
            for (int i = ii; i < min(ii + TILE_SIZE, b_row); i++) {
                for (int j = jj; j < min(jj + TILE_SIZE, b_col); j++) {
                    localA[i][j] = b[i * b_col + j];
                }
            }
        }
    }

// Perform systolic matrix multiply
// local matrices localA and localB have been partitioned in dimensions
// 1 and 2 respectively. local matrix C has been partitioned completely

// This partitioning enables to access MAX_SIZE elements in parallel in
// the local matrices. Because of the mode of access of array elements,
// we are able to perform MAX_SIZE*MAX_SIZE operations in parallel.

// Note : i, j and k loops are interchanged.

// The top loop systolic1 runs only for a_col iterations instead of
// MAX_SIZE like the inner loops. The inner loops have fixed loop
// iteration counts to enable complete unroll

// The following diagram explains how the matrix multiply happens
//
//        B_0        B_1        B_2        B_3
//         |          |          |          |
//         v          v          v          v
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A0_->|C00| ---- |C01| ---- |C02| ---- |C03|
//       |___|      |___|      |___|      |___|
//         |          |          |          |
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A1_->|C10| ---- |C11| ---- |C12| ---- |C13|
//       |___|      |___|      |___|      |___|
//         |          |          |          |
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A2_->|C20| ---- |C21| ---- |C21| ---- |C21|
//       |___|      |___|      |___|      |___|
//         |          |          |          |
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A3_->|C30| ---- |C31| ---- |C32| ---- |C33|
//       |___|      |___|      |___|      |___|

systolic1:
    for (int kk = 0; kk < a_col; kk += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min = DATA_SIZE/TILE_SIZE max = MAX_SIZE/TILE_SIZE
    systolic2:
        for (int ii = 0; ii < MAX_SIZE; ii += TILE_SIZE) {
#pragma HLS UNROLL
        systolic3:
            for (int jj = 0; jj < DATA_SIZE; jj += TILE_SIZE) {
#pragma HLS UNROLL
                // Procesa cada tile
                for (int k = kk; k < min(kk + TILE_SIZE, a_col); k++) {
                    for (int i = ii; i < min(ii + TILE_SIZE, MAX_SIZE); i++) {
                        for (int j = jj; j < min(jj + TILE_SIZE, DATA_SIZE); j++) {
                            // Get previous sum
                            int last = (k == 0) ? 0 : localC[i][j];

                            // Update current sum
                            // Handle boundary conditions
                            int a_val = (i < a_row && k < a_col) ? localA[i][k] : 0;
                            int b_val = (k < b_row && j < b_col) ? localB[k][j] : 0;
                            int result = last + a_val * b_val;

                            // Write back results
                            localC[i][j] = result;
                        }
                    }
                }
            }
        }
    }

// Burst write from output matrices to global memory
// Burst write from matrix C
writeC:
    for (int ii = 0; ii < c_row; ii += TILE_SIZE) {
#pragma HLS LOOP_TRIPCOUNT min = (DATA_SIZE * DATA_SIZE)/TILE_SIZE max = (MAX_SIZE * MAX_SIZE)/TILE_SIZE
        for (int jj = 0; jj < c_col; jj += TILE_SIZE) {
            // Procesa cada tile
            for (int i = ii; i < min(ii + TILE_SIZE, c_row); i++) {
                for (int j = jj; j < min(jj + TILE_SIZE, c_col); j++) {
                    c[i * c_col + j] = localC[i][j];
                }
            }
        }
    }
}
}
