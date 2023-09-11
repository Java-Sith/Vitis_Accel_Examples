AIE Bare-Metal Emulation Test 
=============================

The baremetal flow is used for emulation platform level basic testing i.e. to test different applications - DDR testing, LPDDR testing, AIE testing. 

Input from the user
--------------------

1. The user needs to build PL XOs and libadf.a to be used at v++ link. 
  * Users can also try pre-compiled XOs (fixed XO) and libadf.a (fixed libadf.a) based on their design. Make note to modify the connectivity file (system.cfg) based on the design. 
  * In the Makefile, users can provide pre-compiled XOs for PL kernels as ``PL_O := <fixed XO>`` and for AIE kernels as ``GRAPH_O := <fixed libadf.a>`` 

2. The user needs to generate the fixed XSA from v++ link using extensible XSA.  

Steps to run the Bare-Metal Emulation Test
------------------------------------------

1. Build the fixed xsa using v++ link step as per extensible XSA:  
   ``make fixed_xsa EXTENSIBLE_XSA=$PLATFORM_REPO_PATHS/xilinx_vck190_base_202310_1/hw_emu/hw_emu.xsa``

2. Build the BSP sources and libraries required for compilation of user application. 
   Compile and link the user application to generate main.elf : ``make baremetal_elf``

* Note: Users can modify sw/main.cpp file and incrementally compile it to build the main.elf as per your user application.

3. Generate the package directory: ``make package``

4. Run the user application: ``./package.hw_emu/launch_hw_emu.sh``

* To run all the steps at once, use ``make run TARGET=hw_emu EXTENSIBLE_XSA=$PLATFORM_REPO_PATHS/xilinx_vck190_base_202310_1/hw_emu/hw_emu.xsa``
