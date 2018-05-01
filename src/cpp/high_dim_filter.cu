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

__global__ void  compute_bilateral_kernel_gpu(const  int num_pixels_,
                                        const float * const rgb_blob,
                                        const int width_,
                                        float theta_alpha_, float theta_beta_,
                                        float* const output_kernel) {
    CUDA_KERNEL_LOOP(p, num_pixels_) {
        output_kernel[5 * p] = (float)(p % width_) / theta_alpha_;
        output_kernel[5 * p + 1] = (float)(p / width_) / theta_alpha_;
        output_kernel[5 * p + 2] = (float)(rgb_blob[p] / theta_beta_);
        output_kernel[5 * p + 3] = (float)((rgb_blob + num_pixels_)[p] / theta_beta_);
        output_kernel[5 * p + 4] = (float)((rgb_blob + num_pixels_ * 2)[p] / theta_beta_);
    }
}

__global__ void compute_spatial_kernel_gpu(const  int num_pixels_,
                                        const int width_, float theta_gamma_,
                                        float* const output_kernel) {

  CUDA_KERNEL_LOOP(p, num_pixels_) {
    output_kernel[2*p] = static_cast<float>(p % width_) / theta_gamma_;
    output_kernel[2*p + 1] = static_cast<float>(p / width_) / theta_gamma_;
  }
}

template<>
void HighDimFilterFunctor<GPUDevice>::operator()(
                  const GPUDevice& d, 
                  const tensorflow::Tensor & input_q,
                  const tensorflow::Tensor & input_img,
                  tensorflow::Tensor * out,
                  const FilterParams & params) {

    const int channels = input_q.dim_size(0);
    const int height = input_img.dim_size(1);
    const int width = input_img.dim_size(2);
    const int num_pixels = width * height;

    ModifiedPermutohedral mp;

    if (params.bilateral_) {
      float * kernel_vals;

      CUDA_CHECK(cudaMalloc((void**)&kernel_vals, 5 * num_pixels * sizeof(float)));
      compute_bilateral_kernel_gpu<<<CUDA_GET_BLOCKS(num_pixels), CUDA_NUM_THREADS>>>(
        num_pixels, input_img.flat<float>().data(), width,
        params.theta_alpha_, params.theta_beta_, kernel_vals);
      CUDA_POST_KERNEL_CHECK;

      mp.init_gpu(kernel_vals, 5, width, height);
      mp.compute_gpu(out->flat<float>().data(), input_q.flat<float>().data(), channels, params.backwards_);
      CUDA_CHECK(cudaFree(kernel_vals));
    } else {
      float * kernel_vals;
      CUDA_CHECK(cudaMalloc((void**)&kernel_vals, 2 * num_pixels * sizeof(float)));
      compute_spatial_kernel_gpu<<<CUDA_GET_BLOCKS(num_pixels), CUDA_NUM_THREADS>>>(
        num_pixels, width, params.theta_gamma_, kernel_vals);
      CUDA_POST_KERNEL_CHECK;

      mp.init_gpu(kernel_vals, 2, width, height);
      mp.compute_gpu(out->flat<float>().data(), input_q.flat<float>().data(), channels, params.backwards_);
      CUDA_CHECK(cudaFree(kernel_vals));
    }

    mp.freeMatrix();
 
}


template struct HighDimFilterFunctor<GPUDevice>;

#endif