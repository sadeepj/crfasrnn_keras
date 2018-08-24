#  -----------------------------------------------------------------------------------------------------------------------
#  Instructions:
#
#  1.  Activate your Tensorflow virtualenv before running this script. Before running this script, 'python' command in the
#      terminal should refer to the Python interpreter associated to your Tensorflow installation.
#
#  2.  Run 'make', it should produce a new file named 'high_dim_filter.so'.
#
#  3.  If this script fails, please refer to https://www.tensorflow.org/extend/adding_an_op#build_the_op_library for help.
#
#  -----------------------------------------------------------------------------------------------------------------------

# Define the compiler
CC := g++

# Define the Python version
PYTHON := python3.6

# high_dim_filter.so build dir
BUILD_DIR := build

#Use GPU Implementation?
USE_GPU := 1

# Read Tensorflow paths
TF_INC := $(shell $(PYTHON) -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell $(PYTHON) -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=$(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 

# Is the Tensorflow version >= 1.4?
TF_VERSION_GTE_1_4 := $(shell expr `$(PYTHON) -c 'import tensorflow as tf; print(tf.__version__)' | cut -f1,2 -d.` \>= 1.4)

# Flags required for all cases
CFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC -I$(TF_INC) -O2 

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

# Set some more flags if the Tensorflow version is >= 1.4
ifeq ($(TF_VERSION_GTE_1_4), 1)
    CFLAGS += -I$(TF_INC)/external/nsync/public
	LDFLAGS := -L$(TF_LIB) -ltensorflow_framework -L/usr/local/cuda/lib64 -lcuda -lcudart
else
	LDFLAGS :=
endif

# Define build targets
.PHONY: all clean


all: high_dim_filter.so

high_dim_filter.so: cudacode.o modified_permutohedral.o modified_permutohedral_3d.o high_dim_filter.cc modified_permutohedral.cc 
	g++ $(CFLAGS) -o $(BUILD_DIR)/high_dim_filter.so high_dim_filter.cc modified_permutohedral.cc $(BUILD_DIR)/cudacode.o $(BUILD_DIR)/modified_permutohedral.o $(BUILD_DIR)/modified_permutohedral_3d.o $(LDFLAGS) -D FILTER_GPU=$(USE_GPU) && \
	$(RM) $(BUILD_DIR)/*.o

cudacode.o: high_dim_filter.cu 
	nvcc -std=c++11 -c -o $(BUILD_DIR)/cudacode.o high_dim_filter.cu $(TF_CFLAGS) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D FILTER_GPU=$(USE_GPU)
	
modified_permutohedral_3d.o: modified_permutohedral_3d.cu
	nvcc -std=c++11 -c -o $(BUILD_DIR)/modified_permutohedral_3d.o modified_permutohedral_3d.cu $(TF_CFLAGS) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D FILTER_GPU=$(USE_GPU)

modified_permutohedral.o: modified_permutohedral.cu
	nvcc -std=c++11 -c -o $(BUILD_DIR)/modified_permutohedral.o modified_permutohedral.cu $(TF_CFLAGS) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D FILTER_GPU=$(USE_GPU)

clean:
	$(RM) $(BUILD_DIR)/high_dim_filter.so $(BUILD_DIR)/cudacode.o $(BUILD_DIR)/modified_permutohedral.o $(BUILD_DIR)/modified_permutohedral_3d.o
