#!/usr/bin/env bash

# You may need to modify the following paths before compiling.

# CUDA_HOME=/usr/local/cuda-10.0 \
# CUDNN_INCLUDE_DIR=/usr/local/cuda-10.0/include \
# CUDNN_LIB_DIR=/usr/local/cuda-10.0/lib64 \

export CUDA_PATH=/usr/local/cuda
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
export CUDA_HOME=/usr/local/cuda

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
python setup.py build develop