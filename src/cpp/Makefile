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

# Read Tensorflow paths
TF_INC := $(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# Is the Tensorflow version >= 1.4?
TF_VER_NUM := $(shell expr `${PYTHON} -c 'import tensorflow as tf; print(tf.__version__)'`)
TF_VER_MAJOR := $(shell echo $(TF_VER_NUM) | cut -f1 -d.)
TF_VER_MINOR := $(shell echo $(TF_VER_NUM) | cut -f2 -d.)

TF_VERSION_GTE_1_4 := $(shell [ $(TF_VER_MAJOR) -gt 1 -o \( $(TF_VER_MAJOR) -eq 1 -a $(TF_VER_MINOR) -ge 4 \) ] && echo true)


# Flags required for all cases
CFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC -I$(TF_INC) -O2 

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

# Set some more flags if the Tensorflow version is >= 1.4
ifeq ($(TF_VERSION_GTE_1_4), true)
    CFLAGS += -I$(TF_INC)/external/nsync/public
    LDFLAGS := -L$(TF_LIB) -ltensorflow_framework
else
    LDFLAGS :=
endif

# Define build targets
.PHONY: all clean

high_dim_filter.so: high_dim_filter.cc modified_permutohedral.cc
	$(CC) $(CFLAGS) -o high_dim_filter.so high_dim_filter.cc modified_permutohedral.cc $(LDFLAGS)

clean:
	$(RM) high_dim_filter.so

all: high_dim_filter.so
