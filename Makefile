# make                CPU build (testing), build/fdchan
# make GPU=cuda       NVIDIA (nvcc + cuFFT), set ARCH (default sm_90 for GH200/H100)
# make GPU=hip        AMD (hipcc + hipFFT), set ARCH (default gfx90a for MI250X)
# make SINGLE=1       single precision
# make MPI=1          several ranks (z-slabs): MPICXX (CPU) or nvcc/hipcc with the MPI compile/link flags
# make GPU=cuda MPI=1 NCCL=1  halo exchanges and all-to-alls with NCCL (NVLink); GPU=hip: RCCL.
#                     NCCL_HOME is found next to nvcc in the NVIDIA HPC SDK (tools/find_nccl.sh) unless set
GPU    ?=
MPI    ?= 0
MPICXX ?= $(HOME)/opt/openmpi/bin/mpicxx
ARCH   ?=
SINGLE ?= 0
NCCL   ?= 0
SRC = src/main.cpp
HDR = $(wildcard src/*.hpp)
DEFS =
ifeq ($(SINGLE),1)
DEFS += -DFD_SINGLE
endif
ifeq ($(MPI),1)
DEFS += -DFD_MPI
MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_LFLAGS := $(shell $(MPICXX) --showme:link 2>/dev/null)
endif
# nvcc and hipcc do not accept GCC's -Wl,opt (e.g. the rpath flags of Open MPI); pass them as -Xlinker opt
comma := ,
MPI_LFLAGS_DEV = $(subst -Wl$(comma),-Xlinker ,$(MPI_LFLAGS))
ifeq ($(NCCL),1)
DEFS += -DFD_NCCL
ifeq ($(GPU),cuda)
NCCL_HOME ?= $(shell sh tools/find_nccl.sh)
NCCL_LIB = -lnccl
else
NCCL_LIB = -lrccl
endif
ifneq ($(strip $(NCCL_HOME)),)
NCCL_CFLAGS = -I$(NCCL_HOME)/include
NCCL_LFLAGS = -L$(NCCL_HOME)/lib -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib
endif
endif

ifeq ($(GPU),cuda)
ARCH ?= sm_90
CXX = nvcc
CXXFLAGS = -O3 -std=c++17 -x cu --extended-lambda --expt-relaxed-constexpr -arch=$(if $(ARCH),$(ARCH),sm_90) -DFD_CUDA
LIBS = -lcufft
ifeq ($(MPI),1)
CXXFLAGS += $(MPI_CFLAGS)
LIBS += $(MPI_LFLAGS_DEV)
endif
ifeq ($(NCCL),1)
CXXFLAGS += $(NCCL_CFLAGS)
LIBS += $(NCCL_LFLAGS) $(NCCL_LIB)
endif
else ifeq ($(GPU),hip)
CXX = hipcc
CXXFLAGS = -O3 -std=c++17 --offload-arch=$(if $(ARCH),$(ARCH),gfx90a) -DFD_HIP
LIBS = -lhipfft
ifeq ($(MPI),1)
CXXFLAGS += $(MPI_CFLAGS)
LIBS += $(MPI_LFLAGS_DEV)
endif
ifeq ($(NCCL),1)
CXXFLAGS += $(NCCL_CFLAGS)
LIBS += $(NCCL_LFLAGS) $(NCCL_LIB)
endif
else
ifeq ($(MPI),1)
CXX = $(MPICXX)
else
CXX ?= c++
endif
CXXFLAGS = -O3 -std=c++17 -march=native -Wall -Wextra -Wno-unused-parameter
LIBS = -lpthread
endif
EXE = build/fdchan$(if $(filter 1,$(MPI)),_mpi,)

# rebuild when the compiler or the options change (e.g. NCCL=1 added for an existing EXE), not only the sources
FLAGS_STAMP = build/.flags.$(notdir $(EXE))
BUILD_CMD = $(CXX) $(CXXFLAGS) $(DEFS) $(LIBS)

$(EXE): $(SRC) $(HDR) $(FLAGS_STAMP)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) $(SRC) -o $@ $(LIBS)

$(FLAGS_STAMP): FORCE
	@mkdir -p build
	@echo '$(BUILD_CMD)' | cmp -s - $@ || echo '$(BUILD_CMD)' > $@

FORCE:

# checks across GPUs and nodes (examples/arrhenius_nccl_test.sh):
#   make GPU=cuda NCCL=1 build/nccl_check          NCCL without MPI (tools/nccl_check.cu)
#   make MPI=1 MPICXX=mpicxx build/mpi_check         MPI alone (tools/mpi_check.cpp)
build/nccl_check: tools/nccl_check.cu
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(NCCL_CFLAGS) tools/nccl_check.cu -o $@ $(LIBS) $(NCCL_LFLAGS) $(NCCL_LIB)

build/mpi_check: tools/mpi_check.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) tools/mpi_check.cpp -o $@ $(LIBS)

build/test_zcompact: tests/test_zcompact.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) tests/test_zcompact.cpp -o $@ $(LIBS)

build/test_ops: tests/test_ops.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) tests/test_ops.cpp -o $@ $(LIBS)

build/test_scalars: tests/test_scalars.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) tests/test_scalars.cpp -o $@ $(LIBS)

test: build/test_ops build/test_zcompact build/test_scalars
	./build/test_ops
	./build/test_zcompact
	./build/test_scalars

clean:
	rm -rf build

.PHONY: test clean FORCE
