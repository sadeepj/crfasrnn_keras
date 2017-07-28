#ifndef MODIFIED_PERMUTOHEDRAL_HPP_
#define MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/
class ModifiedPermutohedral {
protected:
  struct Neighbors {
    int n1, n2;

    Neighbors(int n1 = 0, int n2 = 0) : n1(n1), n2(n2) {
    }
  };

  std::vector<int> offset_, rank_;
  std::vector<float> barycentric_;
  std::vector<Neighbors> blur_neighbors_;
  // Number of elements, size of sparse discretized space, dimension of features
  int N_, M_, d_;

  void sseCompute(Tensor &out, const Tensor &in, int value_size,
                  bool reverse = false, bool add = false) const;

  void seqCompute(Tensor &out, const Tensor &in, int value_size,
                  bool reverse = false, bool add = false) const;

public:
  ModifiedPermutohedral();

  void init(const float *features, int num_dimensions, int num_points);

  void compute(Tensor &out, const Tensor &in, int value_size,
               bool reverse = false, bool add = false) const;
};

#endif //_MODIFIED_PERMUTOHEDRAL_HPP_
