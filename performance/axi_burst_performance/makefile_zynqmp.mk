#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#

############################## Help Section ##############################
ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make test PLATFORM=<FPGA platform>"
	$(ECHO) "     Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      EDGE_COMMON_SW is required for SoC shells. Please download and use the pre-built image from - "
	$(ECHO) "      https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html"
	$(ECHO) ""
endif

TARGET := hw
SYSROOT := $(EDGE_COMMON_SW)/sysroots/cortexa72-cortexa53-xilinx-linux
SD_IMAGE_FILE := $(EDGE_COMMON_SW)/Image

include ./utils.mk

TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

# SoC variables
RUN_APP_SCRIPT = ./run_app.sh
PACKAGE_OUT = ./package.$(TARGET)

LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh
RESULT_STRING = TEST PASSED

VPP := v++
VPP_PFLAGS := 
CMD_ARGS = -x1 $(BUILD_DIR)/test_kernel_maxi_256bit.xclbin -x2 $(BUILD_DIR)/test_kernel_maxi_512bit.xclbin
SD_CARD := $(PACKAGE_OUT)

CXXFLAGS += -I$(SYSROOT)/usr/include/xrt -I$(XILINX_VIVADO)/include -Wall -O0 -g -std=c++1y
LDFLAGS += -L$(SYSROOT)/usr/lib -pthread -lxilinxopencl

########################## Checking if PLATFORM in allowlist #######################
PLATFORM_BLOCKLIST += zcu102_base_20 zcu104_base_20 vck zc7 aws-vu9p-f1 samsung u2_ nodma 

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/xcl2
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/cmdparser
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/logger
HOST_SRCS += $(XF_PROJ_ROOT)/common/includes/xcl2/xcl2.cpp $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp ./src/host.cpp 
# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

LDFLAGS += --sysroot=$(SYSROOT)

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps 

EXECUTABLE = ./axi_burst_performance
EMCONFIG_DIR = $(TEMP_DIR)

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/test_kernel_maxi_256bit.xclbin
BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_256bit_1.xo
BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_256bit_2.xo
BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_256bit_3.xo
BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_256bit_4.xo
BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_256bit_5.xo
BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_256bit_6.xo
BINARY_CONTAINERS += $(BUILD_DIR)/test_kernel_maxi_512bit.xclbin
BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_512bit_1.xo
BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_512bit_2.xo
BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_512bit_3.xo
BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_512bit_4.xo
BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_512bit_5.xo
BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS += $(TEMP_DIR)/test_kernel_maxi_512bit_6.xo

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-platform check-device check-vitis $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis check-device $(BINARY_CONTAINERS)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/test_kernel_maxi_256bit_1.xo: src/test_kernel_maxi_256bit_1.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_256bit_1 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_256bit_2.xo: src/test_kernel_maxi_256bit_2.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_256bit_2 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_256bit_3.xo: src/test_kernel_maxi_256bit_3.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_256bit_3 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_256bit_4.xo: src/test_kernel_maxi_256bit_4.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_256bit_4 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_256bit_5.xo: src/test_kernel_maxi_256bit_5.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_256bit_5 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_256bit_6.xo: src/test_kernel_maxi_256bit_6.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_256bit_6 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_512bit_1.xo: src/test_kernel_maxi_512bit_1.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_512bit_1 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_512bit_2.xo: src/test_kernel_maxi_512bit_2.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_512bit_2 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_512bit_3.xo: src/test_kernel_maxi_512bit_3.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_512bit_3 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_512bit_4.xo: src/test_kernel_maxi_512bit_4.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_512bit_4 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_512bit_5.xo: src/test_kernel_maxi_512bit_5.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_512bit_5 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/test_kernel_maxi_512bit_6.xo: src/test_kernel_maxi_512bit_6.cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k test_kernel_maxi_512bit_6 --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(BUILD_DIR)/test_kernel_maxi_256bit.xclbin: $(BINARY_CONTAINER_test_kernel_maxi_256bit_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/test_kernel_maxi_256bit.xclbin' $(+)

$(BUILD_DIR)/test_kernel_maxi_512bit.xclbin: $(BINARY_CONTAINER_test_kernel_maxi_512bit_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/test_kernel_maxi_512bit.xclbin' $(+)

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		$(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(LAUNCH_EMULATOR) -run-app $(RUN_APP_SCRIPT) | tee run_app.log; exit $${PIPESTATUS[0]}
endif


.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(LAUNCH_EMULATOR) -run-app $(RUN_APP_SCRIPT) | tee run_app.log; exit $${PIPESTATUS[0]}
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif


############################## Preparing sdcard ##############################
$(BUILD_DIR)/$(PACKAGE_OUT)/test_kernel_maxi_512bit.xclbin: $(BUILD_DIR)/test_kernel_maxi_256bit.xclbin $(BUILD_DIR)/test_kernel_maxi_512bit.xclbin $(EXECUTABLE)
	$(VPP) -p $(BUILD_DIR)/test_kernel_maxi_512bit.xclbin $(VPP_FLAGS) --package.sd_file $(BUILD_DIR)/test_kernel_maxi_256bit.xclbin --package.out_dir $(PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(SD_IMAGE_FILE) --package.sd_file xrt.ini --package.sd_file $(RUN_APP_SCRIPT) --package.sd_file $(EXECUTABLE) --package.sd_file $(EMCONFIG_DIR)/emconfig.json -o $(BUILD_DIR)/$(PACKAGE_OUT)/test_kernel_maxi_512bit.xclbin

sd_card: $(BINARY_CONTAINERS) $(EXECUTABLE) gen_run_app
	make $(BUILD_DIR)/$(PACKAGE_OUT)/test_kernel_maxi_512bit.xclbin

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin

