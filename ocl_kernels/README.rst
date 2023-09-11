OpenCL Kernel Examples
==================================
This section contains OpenCL Kernel Examples.

**Examples Table :**

.. list-table:: 
  :header-rows: 1

  * - **Example**
    - **Description**
    - **Key Concepts/Keywords**
  * - `cl_array_partition <cl_array_partition>`_
    - This example shows how to use array partitioning to improve performance of a kernel
    - 
      **Key Concepts**

      * `Kernel Optimization <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Optimizing-Data-Movement>`__
      * `Array Partition <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Array-Partitioning>`__
      **Keywords**

      * xcl_array_partition
      * `complete <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition>`__

  * - `cl_burst_rw <cl_burst_rw>`_
    - This is simple example of using AXI4-master interface for burst read and write
    - 
      **Key Concepts**

      * `burst access <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/AXI-Burst-Transfers>`__
      **Keywords**

      * `compiler.interfaceRdBurstLen <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/advanced-Options>`__
      * `compiler.interfaceWrBurstLen <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/advanced-Options>`__

  * - `cl_dataflow_func <cl_dataflow_func>`_
    - This is simple example of vector addition to demonstrate Dataflow functionality in OpenCL Kernel. OpenCL Dataflow allows user to run multiple functions together to achieve higher throughput.
    - 
      **Key Concepts**

      * `Function/Task Level Parallelism <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Kernel-Execution>`__
      **Keywords**

      * xcl_dataflow
      * `xclDataflowFifoDepth <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/advanced-Options>`__

  * - `cl_dataflow_subfunc <cl_dataflow_subfunc>`_
    - This is simple example of vector addition to demonstrate how OpenCL Dataflow allows user to run multiple sub functions together to achieve higher throughput.
    - 
      **Key Concepts**

      * SubFunction Level Parallelism

      **Keywords**

      * xcl_dataflow
      * `xclDataflowFifoDepth <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/advanced-Options>`__

  * - `cl_gmem_2banks <cl_gmem_2banks>`_
    - This example of 2ddr to demonstrate on how to use 2ddr XSA. How to create buffers in each DDR.
    - 
      **Key Concepts**

      * Multiple Banks

      **Keywords**

      * `sp <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/connectivity-Options>`__
      * `connectivity <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/connectivity-Options>`__

  * - `cl_helloworld <cl_helloworld>`_
    - This example is a simple OpenCL application. It will highlight the basic flow of an OpenCL application.
    - 
      **Key Concepts**

      * `OpenCL Host APIs <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/OpenCL-Programming>`__

  * - `cl_lmem_2rw <cl_lmem_2rw>`_
    - This is simple example of vector addition to demonstrate how to utilized both ports of Local Memory.
    - 
      **Key Concepts**

      * `Kernel Optimization <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Optimizing-Data-Movement>`__
      * `2port BRAM Utilization <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Port-Level-I/O-Memory-Interface-Protocol>`__
      * two read/write Local Memory

      **Keywords**

      * opencl_unroll_hint

  * - `cl_loop_reorder <cl_loop_reorder>`_
    - This is a simple example of matrix multiplication (Row x Col) to demonstrate how to achieve better pipeline II factor by loop reordering.
    - 
      **Key Concepts**

      * `Kernel Optimization <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Optimizing-Data-Movement>`__
      * Loop reorder to improve II

      **Keywords**

      * xcl_pipeline_loop
      * xcl_array_partition

  * - `cl_partition_cyclicblock <cl_partition_cyclicblock>`_
    - This example shows how to use array block and cyclic partitioning to improve performance of a kernel
    - 
      **Key Concepts**

      * `Kernel Optimization <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Optimizing-Data-Movement>`__
      * Array Partitioning

      * `Block Partition <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition>`__
      * `Cyclic Partition <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition>`__
      **Keywords**

      * xcl_array_partition
      * `cyclic <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition>`__
      * `block <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition>`__

  * - `cl_shift_register <cl_shift_register>`_
    - This example demonstrates how to shift values in registers in each clock cycle
    - 
      **Key Concepts**

      * `Kernel Optimization <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Optimizing-Data-Movement>`__
      * `Shift Register <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Inferring-Shift-Registers>`__
      * `FIR <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Inferring-Shift-Registers>`__
      **Keywords**

      * xcl_array_partition
      * getProfilingInfo

  * - `cl_systolic_array <cl_systolic_array>`_
    - This is a simple example of matrix multiplication (Row x Col) to help developers learn systolic array based algorithm design. Note: Systolic array based algorithm design is well suited for FPGA.
    - 

  * - `cl_wide_mem_rw <cl_wide_mem_rw>`_
    - This is simple example of vector addition to demonstrate Wide Memory Access using uint16 data type. Based on input argument type, V++ compiler will figure our the memory datawidth between Global Memory and Kernel. For this example, uint16 datatype is used, so Memory datawidth will be 16 x (integer bit size) = 16 x 32 = 512 bit.
    - 
      **Key Concepts**

      * `Kernel to DDR <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Mapping-Kernel-Ports-to-Memory>`__
      * `wide memory access <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/AXI-Burst-Transfers>`__
      * `burst read and write <https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/AXI-Burst-Transfers>`__
      **Keywords**

      * uint16
      * xcl_pipeline_loop


