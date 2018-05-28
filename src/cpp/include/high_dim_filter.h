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

#ifndef KERNEL_HIGH_DIM_FILTER_H_
#define KERNEL_HIGH_DIM_FILTER_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "modified_permutohedral.h"


using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

struct FilterParams
{
  FilterParams(bool bilateral, 
               float theta_alpha, 
               float theta_beta, 
               float theta_gamma, 
               bool backwards) : 
               bilateral_(bilateral), 
               theta_alpha_(theta_alpha), 
               theta_beta_(theta_beta), 
               theta_gamma_(theta_gamma), 
               backwards_(backwards) {}

   bool bilateral_;
   float theta_alpha_;
   float theta_beta_;
   float theta_gamma_;
   bool backwards_;
};

template <typename Device>
struct HighDimFilterFunctor {
  void operator()(const Device& d, 
                  const tensorflow::Tensor & input_q,
                  const tensorflow::Tensor & input_img,
                  tensorflow::Tensor * out,
                  const FilterParams & params);
};


#endif // KERNEL_HIGH_DIM_FILTER_H_