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