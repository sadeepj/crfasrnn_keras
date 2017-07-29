/*
 *  MIT License
 *
 *  Copyright (c) 2017 Sadeep Jayasumana
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
#include "modified_permutohedral.h"

using namespace tensorflow;

void compute_spatial_kernel(float * const output_kernel, const int width,
                            const int height, const float theta_gamma) {

  const int num_pixels = width * height;
  for (int p = 0; p < num_pixels; ++p) {
    output_kernel[2 * p] = static_cast<float>(p % width) / theta_gamma;
    output_kernel[2 * p + 1] = static_cast<float>(p / width) / theta_gamma;
  }
}

void compute_bilateral_kernel(float * const output_kernel, const Tensor& rgb_tensor,
                              const float theta_alpha, const float theta_beta) {

  const int height = rgb_tensor.dim_size(1);
  const int width = rgb_tensor.dim_size(2);
  const int num_pixels = height * width;
  auto rgb = rgb_tensor.flat<float>();

  for (int p = 0; p < num_pixels; ++p) {
    // Spatial terms
    output_kernel[5 * p] = static_cast<float>(p % width) / theta_alpha;
    output_kernel[5 * p + 1] = static_cast<float>(p / width) / theta_alpha;

    // Color terms
    output_kernel[5 * p + 2] = static_cast<float>(rgb(p) / theta_beta);
    output_kernel[5 * p + 3] = static_cast<float>(rgb(num_pixels + p) / theta_beta);
    output_kernel[5 * p + 4] = static_cast<float>(rgb(2 * num_pixels + p) / theta_beta);
  }
}

REGISTER_OP("HighDimFilter")
    .Attr("bilateral: bool")
    .Attr("theta_alpha: float = 1.0")
    .Attr("theta_beta: float = 1.0")
    .Attr("theta_gamma: float = 1.0")
    .Attr("backwards: bool = false")
    .Input("raw: float32")
    .Input("rgb: float32")
    .Output("filtered: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class HighDimFilterOp : public OpKernel {
 public:
  explicit HighDimFilterOp(OpKernelConstruction* context) : OpKernel(context) {

    OP_REQUIRES_OK(context,
                   context->GetAttr("bilateral", &bilateral_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_alpha", &theta_alpha_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_beta", &theta_beta_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("theta_gamma", &theta_gamma_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("backwards", &backwards_));
  }

  void Compute(OpKernelContext* context) override {

    // Grab the unary tensor
    const Tensor& input_tensor = context->input(0);
    // Grab the RGB image tensor
    const Tensor& image_tensor = context->input(1);

    const int channels = input_tensor.dim_size(0);
    const int height = input_tensor.dim_size(1);
    const int width = input_tensor.dim_size(2);
    const int num_pixels = width * height;

    // Create the output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    ModifiedPermutohedral mp;

    if (bilateral_) {
      float * const kernel_vals = new float[5 * num_pixels];
      compute_bilateral_kernel(kernel_vals, image_tensor,
                               theta_alpha_, theta_beta_);
      mp.init(kernel_vals, 5, num_pixels);
      mp.compute(*output_tensor, input_tensor, channels, backwards_);

      delete[] kernel_vals;
    } else {
      float * const kernel_vals = new float[2 * num_pixels];
      compute_spatial_kernel(kernel_vals, width, height, theta_gamma_);
      mp.init(kernel_vals, 2, num_pixels);
      mp.compute(*output_tensor, input_tensor, channels, backwards_);

      delete[] kernel_vals;
    }

  }
 
 private:
  bool bilateral_;
  float theta_alpha_;
  float theta_beta_;
  float theta_gamma_;
  bool backwards_;
};

REGISTER_KERNEL_BUILDER(Name("HighDimFilter").Device(DEVICE_CPU), HighDimFilterOp);
