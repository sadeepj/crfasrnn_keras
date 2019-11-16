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
PYTHON=python
CC := g++

# Read Tensorflow flags 
TF_CFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell ${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	MACFLAGS := -undefined dynamic_lookup
else
	MACFLAGS :=
endif

# Define build targets
.PHONY: all clean

high_dim_filter.so: high_dim_filter.cc modified_permutohedral.cc
	$(CC) -std=c++11 -shared high_dim_filter.cc modified_permutohedral.cc -o high_dim_filter.so -fPIC $(TF_CFLAGS) $(MACFLAGS) $(TF_LFLAGS) -O2

clean:
	$(RM) high_dim_filter.so

all: high_dim_filter.so
