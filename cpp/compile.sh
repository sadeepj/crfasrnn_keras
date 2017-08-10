#  ------------------------------------------------------------------------------------------------------------i----------
#  *  Activate your Tensorflow virtualenv before running this script.
#  *  This script assumes gcc version >=5. If you have an older version, remove the -D_GLIBCXX_USE_CXX11_ABI=0 flag below.
#  *  On Mac OS X, the additional flag "-undefined dynamic_lookup" is required.
#  *  If this script fails, please refer to https://www.tensorflow.org/extend/adding_an_op#build_the_op_library for help.
#  -----------------------------------------------------------------------------------------------------------------------

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared high_dim_filter.cc modified_permutohedral.cc -o high_dim_filter.so -fPIC -I $TF_INC -O2
