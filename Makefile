# make                CPU build (testing), build/fdchan
# make GPU=cuda       NVIDIA (nvcc + cuFFT), set ARCH (default sm_90 for GH200/H100)
# make GPU=hip        AMD (hipcc + hipFFT), set ARCH (default gfx90a for MI250X)
# make SINGLE=1       single precision
GPU    ?=
ARCH   ?=
SINGLE ?= 0
SRC = src/main.cpp
HDR = $(wildcard src/*.hpp)
DEFS =
ifeq ($(SINGLE),1)
DEFS += -DFD_SINGLE
endif

ifeq ($(GPU),cuda)
ARCH ?= sm_90
CXX = nvcc
CXXFLAGS = -O3 -std=c++17 -x cu --extended-lambda --expt-relaxed-constexpr -arch=$(if $(ARCH),$(ARCH),sm_90) -DFD_CUDA
LIBS = -lcufft
else ifeq ($(GPU),hip)
CXX = hipcc
CXXFLAGS = -O3 -std=c++17 --offload-arch=$(if $(ARCH),$(ARCH),gfx90a) -DFD_HIP
LIBS = -lhipfft
else
CXX ?= c++
CXXFLAGS = -O3 -std=c++17 -march=native -Wall -Wextra -Wno-unused-parameter
LIBS = -lpthread
endif

build/fdchan: $(SRC) $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) $(SRC) -o $@ $(LIBS)

build/test_ops: tests/test_ops.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(DEFS) tests/test_ops.cpp -o $@ $(LIBS)

test: build/test_ops
	./build/test_ops

clean:
	rm -rf build

.PHONY: test clean
