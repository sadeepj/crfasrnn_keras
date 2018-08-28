/*
 *  MIT License
 *
 *  Copyright (c) 2017 Sadeep Jayasumana
 *
 *  Author Tom Joy 2018, tomjoy@robots.ox.ac.uk
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:

 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.

 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#if FILTER_GPU
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "include/high_dim_filter.h"
#include "include/cuda_macros.h"


using GPUDevice = Eigen::GpuDevice;
using namespace tensorflow;

__global__ void  compute_bilateral_kernel_gpu(const float * const rgb_blob,
                                              const int num_pixels_,
                                              const int bilateral_channels_,
                                              const int width_,
                                              float theta_alpha_,
                                              float theta_beta_,
                                              float* const output_kernel) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_pixels_) return;

  const int bc = bilateral_channels_;
  output_kernel[bc * p] = (float)(p % width_) / theta_alpha_;
  output_kernel[bc * p + 1] = (float)(p / width_) / theta_alpha_;
  for (int i = 2; i < bc; ++i) {
    int dnp = i - 2;
    output_kernel[bc * p + i] = (float)(rgb_blob[p + dnp * num_pixels_] / theta_beta_);
  }
}

__global__ void  compute_bilateral_kernel_3d_gpu(const float * const rgb_blob,
                                                 const int num_pixels_,
                                                 const int bilateral_channels_,
                                                 const int height_,
                                                 const int width_,
                                                 float theta_alpha_,
                                                 float theta_alpha_z_,
                                                 float theta_beta_,
                                                 float* const output_kernel) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_pixels_) return;

  const int bc = bilateral_channels_;
  const int hw = height_ * width_;
  output_kernel[bc * p] = (float)(p % width_) / theta_alpha_;
  output_kernel[bc * p + 1] = (float)(p / width_) / theta_alpha_;
  output_kernel[bc * p + 2] = (float)(p / hw) / theta_alpha_z_;
  for (int i = 3; i < bc; ++i) {
    int dnp = i - 3;
    output_kernel[bc * p + i] = (float)(rgb_blob[p + dnp * num_pixels_] / theta_beta_);
  }
}

__global__ void compute_spatial_kernel_gpu(const int num_pixels_,
                                           const int width_,
                                           float theta_gamma_,
                                           float* const output_kernel) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_pixels_) return;

  output_kernel[2 * p] = (float)(p % width_) / theta_gamma_;
  output_kernel[2 * p + 1] = (float)(p / width_) / theta_gamma_;  
}

__global__ void compute_spatial_kernel_3d_gpu(const int num_pixels_,
                                              const int height_,
                                              const int width_,
                                              float theta_gamma_,
                                              float theta_gamma_z_,
                                              float* const output_kernel) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_pixels_) return;

  const int hw = height_ * width_;
  output_kernel[3 * p] = (float)(p % width_) / theta_gamma_;
  output_kernel[3 * p + 1] = (float)(p / width_) / theta_gamma_;
  output_kernel[3 * p + 2] = (float)(p / hw) / theta_gamma_z_;
}

template<>
void HighDimFilterFunctor<GPUDevice>::operator()(
                  const GPUDevice& d, 
                  const tensorflow::Tensor & unary_tensor,
                  const tensorflow::Tensor & image_tensor,
                  tensorflow::Tensor * out,
                  const FilterParams & params) {
    const int spatial_dims = image_tensor.dims() - 1;
    const bool is_3d = spatial_dims == 3;

    const int image_channels = image_tensor.dim_size(0);
    const int bilateral_channels = image_channels + spatial_dims;
    const int unary_channels = unary_tensor.dim_size(0);
    const int depth = is_3d ? image_tensor.dim_size(1): 1;
    const int height = image_tensor.dim_size(spatial_dims - 1);
    const int width = image_tensor.dim_size(spatial_dims);
    const int num_pixels = width * height * depth;

    float * kernel_vals;
    if (is_3d) {
      // 3D spatial layout for data
      ModifiedPermutohedral3D mp;

      if (params.bilateral_) {
        // Bilateral filter kernel computation
        int kernel_vals_size = bilateral_channels * num_pixels * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&kernel_vals, kernel_vals_size));

        compute_bilateral_kernel_3d_gpu<<<CUDA_GET_BLOCKS(num_pixels), 
                                          CUDA_NUM_THREADS>>>(
          image_tensor.flat<float>().data(),
          num_pixels,
          bilateral_channels,
          height,
          width,
          params.theta_alpha_,
          params.theta_alpha_z_,
          params.theta_beta_,
          kernel_vals);
        CUDA_POST_KERNEL_CHECK;
        mp.init_gpu(kernel_vals, bilateral_channels, width, height, depth);
        mp.compute_gpu(out->flat<float>().data(),
                       unary_tensor.flat<float>().data(),
                       unary_channels,
                       params.backwards_);
        CUDA_CHECK(cudaFree(kernel_vals));
      
      } else {
        // Spatial filter kernel computation
        int kernel_vals_size = spatial_dims * num_pixels * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&kernel_vals, kernel_vals_size));

        compute_spatial_kernel_3d_gpu<<<CUDA_GET_BLOCKS(num_pixels), 
                                        CUDA_NUM_THREADS>>>(
          num_pixels, 
          height,
          width, 
          params.theta_gamma_, 
          params.theta_gamma_z_,
          kernel_vals);
        CUDA_POST_KERNEL_CHECK;
        mp.init_gpu(kernel_vals, spatial_dims, width, height, depth);
        mp.compute_gpu(out->flat<float>().data(),
                       unary_tensor.flat<float>().data(),
                       unary_channels,
                       params.backwards_);
        CUDA_CHECK(cudaFree(kernel_vals));
      }
      mp.freeMatrix();
    
    } else {
      // 2D spatial layout for data
      ModifiedPermutohedral mp;

      if (params.bilateral_) {
        // Bilateral filter kernel computation
        int kernel_vals_size = bilateral_channels * num_pixels * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&kernel_vals, kernel_vals_size));

        compute_bilateral_kernel_gpu<<<CUDA_GET_BLOCKS(num_pixels), 
                                     CUDA_NUM_THREADS>>>(
          image_tensor.flat<float>().data(),
          num_pixels,
          bilateral_channels,
          width,
          params.theta_alpha_,
          params.theta_beta_,
          kernel_vals);
        CUDA_POST_KERNEL_CHECK;
        mp.init_gpu(kernel_vals, bilateral_channels, width, height);
        mp.compute_gpu(out->flat<float>().data(),
                       unary_tensor.flat<float>().data(),
                       unary_channels,
                       params.backwards_);
        CUDA_CHECK(cudaFree(kernel_vals));
      
      } else {
        // Spatial filter kernel computation
        int kernel_vals_size = spatial_dims * num_pixels * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&kernel_vals, kernel_vals_size));

        compute_spatial_kernel_gpu<<<CUDA_GET_BLOCKS(num_pixels), 
                                        CUDA_NUM_THREADS>>>(
          num_pixels, 
          width, 
          params.theta_gamma_, 
          kernel_vals);
        CUDA_POST_KERNEL_CHECK;
        mp.init_gpu(kernel_vals, spatial_dims, width, height);
        mp.compute_gpu(out->flat<float>().data(),
                       unary_tensor.flat<float>().data(),
                       unary_channels,
                       params.backwards_);
        CUDA_CHECK(cudaFree(kernel_vals));
      }
      mp.freeMatrix();      

    } 
}


template struct HighDimFilterFunctor<GPUDevice>;

#endif
