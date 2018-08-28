/*
 *  MIT License
 *
 *  Copyright (c) 2017 Sadeep Jayasumana
 *
 *  Modified by Tom Joy 2018, tomjoy@robots.ox.ac.uk
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "include/high_dim_filter.h"

using namespace tensorflow;

void compute_spatial_kernel(float * const output_kernel,
                            const int width,
                            const int height,
                            const float theta_gamma) {

    const int num_pixels = width * height;
    for (int p = 0; p < num_pixels; ++p) {
        output_kernel[2 * p] = static_cast<float>(p % width) / theta_gamma;
        output_kernel[2 * p + 1] = static_cast<float>(p / width) / theta_gamma;
    }
}

void compute_spatial_kernel_3d(float * const output_kernel,
                               const int width,
                               const int height,
                               const int depth,
                               const float theta_gamma,
                               const float theta_gamma_z) {
    const int hw = height * width;
    const int num_voxels = depth * height * width;
    for (int p = 0; p < num_voxels; ++p) {
        output_kernel[3 * p] = static_cast<float>(p % width) / theta_gamma;
        output_kernel[3 * p + 1] = static_cast<float>(p / width) / theta_gamma;
        output_kernel[3 * p + 2] = static_cast<float>(p / hw) / theta_gamma_z;
    }
}

void compute_bilateral_kernel(float * const output_kernel,
                              const Tensor& image_tensor,
                              const float theta_alpha,
                              const float theta_beta) {

    const int unary_channels = image_tensor.dim_size(0);
    const int height = image_tensor.dim_size(1);
    const int width = image_tensor.dim_size(2);
    const int num_pixels = height * width;
    auto rgb = image_tensor.flat<float>();

    // Number of output unary_channels: rgb unary_channels plus two spatial (x, y) unary_channels
    const int oc = unary_channels + 2;
    for (int p = 0; p < num_pixels; ++p) {
        // Spatial terms
        output_kernel[oc * p] = static_cast<float>(p % width) / theta_alpha;
        output_kernel[oc * p + 1] = static_cast<float>(p / width) / theta_alpha;

        // Color channel terms
        for (int i = 0; i < unary_channels; ++i) {
            output_kernel[oc * p + i + 2] =
                    static_cast<float>(rgb(p + i * num_pixels) / theta_beta);
        }
    }
}

void compute_bilateral_kernel_3d(float * const output_kernel,
                                 const Tensor& image_tensor,
                                 const float theta_alpha,
                                 const float theta_alpha_z,
                                 const float theta_beta) {
    const int unary_channels = image_tensor.dim_size(0);
    const int depth = image_tensor.dim_size(1);
    const int height = image_tensor.dim_size(2);
    const int width = image_tensor.dim_size(3);
    const int hw = height * width;
    const int num_pixels = depth * height * width;

    auto rgb = image_tensor.flat<float>();

    const int oc = unary_channels + 3;
    for (int p = 0; p < num_pixels; ++p) {
        output_kernel[oc * p] = static_cast<float>(p % width) / theta_alpha;
        output_kernel[oc * p + 1] = static_cast<float>(p / width) / theta_alpha;
        output_kernel[oc * p + 2] = static_cast<float>(p / hw) / theta_alpha_z;

        // Color channel terms
        for (int i = 0; i < unary_channels; ++i) {
            output_kernel[oc * p + i + 3] =
                    static_cast<float>(rgb(p + i * num_pixels)) / theta_beta;
        }
    }
}


REGISTER_OP("HighDimFilter")
    .Attr("T: {float}")
    .Attr("bilateral: bool")
    .Attr("theta_alpha: float = 1.0")
    .Attr("theta_alpha_z: float = 1.0")
    .Attr("theta_beta: float = 1.0")
    .Attr("theta_gamma: float = 1.0")
    .Attr("theta_gamma_z: float = 1.0")
    .Attr("backwards: bool = false")
    .Input("raw: T")
    .Input("rgb: T")
    .Output("filtered: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <>
struct HighDimFilterFunctor<CPUDevice> {
  void operator()(const CPUDevice& d, 
                  const tensorflow::Tensor & unary_tensor,
                  const tensorflow::Tensor & image_tensor,
                  tensorflow::Tensor * out,
                  const FilterParams & params) {
    
    ModifiedPermutohedral mp;

    const int spatial_dims = image_tensor.dims() - 1;
    const bool is_3d = spatial_dims == 3;

    const int image_channels = image_tensor.dim_size(0);
    const int bilateral_channels = image_channels + spatial_dims;
    const int unary_channels = unary_tensor.dim_size(0);
    const int depth = is_3d ? image_tensor.dim_size(1) : 1;
    const int height = image_tensor.dim_size(spatial_dims - 1);
    const int width = image_tensor.dim_size(spatial_dims);
    const int num_pixels = width * height * depth;

    if (params.bilateral_) {
      float * const kernel_vals = new float[bilateral_channels * num_pixels];
      if (is_3d) {
        compute_bilateral_kernel_3d(kernel_vals,
                                    image_tensor,
                                    params.theta_alpha_,
                                    params.theta_alpha_z_,
                                    params.theta_beta_);
      } else {
        compute_bilateral_kernel(kernel_vals,
                                 image_tensor,
                                 params.theta_alpha_,
                                 params.theta_beta_);
      }
      mp.init_cpu(kernel_vals, bilateral_channels, num_pixels);
      mp.compute_cpu(*out, unary_tensor, unary_channels, params.backwards_);

      delete[] kernel_vals;
    } else {
      float * const kernel_vals = new float[spatial_dims * num_pixels];
      if (is_3d) {
        compute_spatial_kernel_3d(kernel_vals,
                                  width,
                                  height,
                                  depth,
                                  params.theta_gamma_,
                                  params.theta_gamma_z_);
      } else {
        compute_spatial_kernel(kernel_vals,
                               width,
                               height,
                               params.theta_gamma_);
      }
      mp.init_cpu(kernel_vals, spatial_dims, num_pixels);
      mp.compute_cpu(*out, unary_tensor, unary_channels, params.backwards_);

      delete[] kernel_vals;
    }
  
  }

};


template <typename Device, typename T>
class HighDimFilterOp : public OpKernel {
 public:
  explicit HighDimFilterOp(OpKernelConstruction* context) : OpKernel(context) {

    OP_REQUIRES_OK(context,
                   context->GetAttr("bilateral", &bilateral_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_alpha", &theta_alpha_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_alpha_z", &theta_alpha_z_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_beta", &theta_beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_gamma", &theta_gamma_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_gamma_z", &theta_gamma_z_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("backwards", &backwards_));
  }

  void Compute(OpKernelContext* context) override {

    // Grab the unary tensor
    const Tensor& unary_tensor = context->input(0);
    // Grab the RGB image tensor
    const Tensor& image_tensor = context->input(1);

    // Create the output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, unary_tensor.shape(),
                                                     &output_tensor));

    const FilterParams params(bilateral_, 
                        theta_alpha_,
                        theta_alpha_z_,
                        theta_beta_, 
                        theta_gamma_,
                        theta_gamma_z_,
                        backwards_);
    //filter
    HighDimFilterFunctor<Device>()(
                   context->eigen_device<Device>(),
                   unary_tensor,
                   image_tensor,
                   output_tensor,
                   params);
  }

 private:
  bool bilateral_;
  float theta_alpha_;
  float theta_alpha_z_;
  float theta_beta_;
  float theta_gamma_;
  float theta_gamma_z_;
  bool backwards_;
};


// Register kernels
#define REGISTER_CPU(T)                         \
  REGISTER_KERNEL_BUILDER(                      \
      Name("HighDimFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      HighDimFilterOp<CPUDevice, T>);
REGISTER_CPU(float);

#if FILTER_GPU
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("HighDimFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      HighDimFilterOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif

