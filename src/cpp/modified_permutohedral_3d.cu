//Modification to the file modified_permutohedral.h from torrvision/crfasrnn

#define BLOCK_SIZE_XY 8
#define BLOCK_SIZE_Z 4
#define CLEAN_BLOCK_SIZE 32

#include <stdio.h>
#include "include/modified_permutohedral_3d.h"
#include "include/cuda_macros.h"
#include "hash_helper.cu"


const int BLOCK_SIZE = BLOCK_SIZE_XY * BLOCK_SIZE_XY * BLOCK_SIZE_Z;

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void gpu_set<int>(const int N, const int alpha, int* Y);
template void gpu_set<float>(const int N, const float alpha, float* Y);
template void gpu_set<double>(const int N, const double alpha, double* Y);

static void swapHashTableValues(float* oldValues, float *newValues, float* table_values,size_t size) {
    CUDA_CHECK(cudaMemcpy(oldValues,table_values,size,cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(table_values,newValues,size,cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(newValues,oldValues,size,cudaMemcpyDeviceToDevice));
    // Works but give poorer results
    //oldValues = table_values;
    //table_values = newValues;
    //newValues = oldValues;
}

template<int pd>
__global__ static void createMatrix(const int w, 
                                    const int h,
                                    const int d,
                                    const float *positions,
                                    int *table_entries,
                                    int table_capacity,
                                    signed short* table_keys,
                                    const float *scaleFactor,
                                    MatrixEntry3D *matrix)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    const int threadId = threadIdx.z * blockDim.x * blockDim.y 
                          + threadIdx.y * blockDim.x 
                          + threadIdx.x;
    const int idx = z * w * h + y * w + x;
    // TODO Keep searching for errors from here
    const bool outOfBounds = (x >= w) || (y >= h) || (z >= d);

    float myElevated[pd+1];
    const float *myPosition = positions + idx*pd;

    int myGreedy[pd+1];
    int myRank[pd+1];

    float myBarycentric[pd+2];
    __shared__ short keys[pd*BLOCK_SIZE];
    short *myKey = keys + threadId * pd;

    if (!outOfBounds) {

        myElevated[pd] = -pd*(myPosition[pd-1])*scaleFactor[pd-1];
        for (int i = pd-1; i > 0; i--) {
            myElevated[i] = (myElevated[i+1] -
                             i*(myPosition[i-1])*scaleFactor[i-1] +
                             (i+2)*(myPosition[i])*scaleFactor[i]);
        }
        myElevated[0] = myElevated[1] + 2*(myPosition[0])*scaleFactor[0];


        // find the closest zero-colored lattice point

        // greedily search for the closest zero-colored lattice point
        signed short sum = 0;
        for (int i = 0; i <= pd; i++) {
            float v = myElevated[i]*(1.0f/(pd+1));
            float up = ceilf(v) * (pd+1);
            float down = floorf(v) * (pd+1);
            if (up - myElevated[i] < myElevated[i] - down) {
                myGreedy[i] = (signed short)up;
            } else {
                myGreedy[i] = (signed short)down;
            }
            sum += myGreedy[i];
        }
        sum /= pd+1;

        // sort differential to find the permutation between this simplex and the canonical one
        for (int i = 0; i <= pd; i++) {
            myRank[i] = 0;
            for (int j = 0; j <= pd; j++) {
                if (myElevated[i] - myGreedy[i] < myElevated[j] - myGreedy[j] ||
                    (myElevated[i] - myGreedy[i] == myElevated[j] - myGreedy[j]
                     && i > j)) {
                    myRank[i]++;
                }
            }
        }

        if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
            for (int i = 0; i <= pd; i++) {
                if (myRank[i] >= pd + 1 - sum) {
                    myGreedy[i] -= pd+1;
                    myRank[i] += sum - (pd+1);
                } else {
                    myRank[i] += sum;
                }
            }
        } else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
            for (int i = 0; i <= pd; i++) {
                if (myRank[i] < -sum) {
                    myGreedy[i] += pd+1;
                    myRank[i] += (pd+1) + sum;
                } else {
                    myRank[i] += sum;
                }
            }
        }

        // turn delta into barycentric coords
        for (int i = 0; i <= pd+1; i++) {
            myBarycentric[i] = 0;
        }

        for (int i = 0; i <= pd; i++) {
            float delta = (myElevated[i] - myGreedy[i]) * (1.0f/(pd+1));
            myBarycentric[pd-myRank[i]] += delta;
            myBarycentric[pd+1-myRank[i]] -= delta;
        }
        myBarycentric[0] += 1.0f + myBarycentric[pd+1];
    }

    for (int color = 0; color <= pd; color++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        if (!outOfBounds) {
            for (int i = 0; i < pd; i++) {
                myKey[i] = myGreedy[i] + color;
                if (myRank[i] > pd-color) myKey[i] -= (pd+1);
            }
        }

        if (!outOfBounds) {
            MatrixEntry3D r;
            r.index = hashTableInsert<pd>(myKey, 
                                          table_keys, 
                                          table_entries,
                                          table_capacity,  
                                          idx*(pd+1)+color);
            r.weight = myBarycentric[color];
            matrix[idx*(pd+1) + color] = r;
        }
    }
}

template<int kd>
__global__ static void cleanHashTable(const int n,
                                      int *table_entries,
                                      int table_capacity,
                                      signed short* table_keys)
{
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // find my hash table entry
    int *e = table_entries + idx;

    // Check if I created my own key in the previous phase
    if (*e >= 0) {
        // Rehash my key and reset the pointer in order to merge with
        // any other pixel that created a different entry under the
        // same key. If the computation was serial this would never
        // happen, but sometimes race conditions can make the same key
        // be inserted twice. hashTableRetrieve always returns the
        // earlier, so it's no problem as long as we rehash now.
        *e = hashTableRetrieve<kd>(table_keys + *e*kd,
                                   table_entries, table_keys, table_capacity);
    }
}

template<int pd>
__global__ static void resetIndex(const int w, 
                                  const int h,
                                  const int d,
                                  MatrixEntry3D *matrix,
                                  int *table_entries)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + (blockIdx.z / (pd + 1)) * blockDim.z;
    const int color = blockIdx.z % (pd + 1);
    const int idx = z * h * w + y * w + x;
    const bool outOfBounds = (x >= w) || (y >= h) || (z >= d);
    if (!outOfBounds){
        MatrixEntry3D r = matrix[idx * (pd + 1) + color];
        matrix[idx*(pd+1)+color].index = table_entries[r.index];
    }
}

template<int pd, typename Dtype>
__global__ static void splatCache(const int w, 
                                  const int h, 
                                  const int d,
                                  const int vd,
                                  const Dtype *values,
                                  const MatrixEntry3D *matrix,
                                  float *table_values)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + (blockIdx.z / (pd + 1)) * blockDim.z;
    const int threadId = threadIdx.z * blockDim.x * blockDim.y 
                          + threadIdx.y * blockDim.x 
                          + threadIdx.x;
    const int color = blockIdx.z % (pd + 1);
    const int idx = z * h * w + y * w + x;
    const bool outOfBounds = (x >= w) || (y >= h) || (z >= d);

    __shared__ int sharedOffsets[BLOCK_SIZE];
    extern __shared__ float sharedValues[];
    int myOffset = -1;
    float *myValue = sharedValues + threadId * (vd + 1);

    if (!outOfBounds) {

        const Dtype *value = values + idx;

        MatrixEntry3D r = matrix[idx * (pd + 1) + color];

        // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
        //matrix[idx*(pd+1)+color].index = r.index = table_entries[r.index];
        // record the offset into the keys/values array in shared space
        myOffset = sharedOffsets[threadId] = r.index * (vd + 1);

        for (int j = 0; j < vd; j++) {
            myValue[j] = (float)value[j * w * h * d] * r.weight;
        }
        myValue[vd] = r.weight;

    } else {
        sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    // am I the first thread in this block to care about this key?

    if (outOfBounds) return;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {
            if (myOffset == sharedOffsets[i]) {
                // somebody else with higher priority cares about this key
                return;
            }
        } else if (i > threadId) {
            if (myOffset == sharedOffsets[i]) {
                // someone else with lower priority cares about this key, accumulate it into mine
                for (int j = 0; j <= vd; j++) {
                    sharedValues[threadId * (vd+1) + j] += 
                        sharedValues[i * (vd + 1) + j];
                }
            }
        }
    }

    // only the threads with something to write to main memory are still going
    float *val = table_values + myOffset;
    for (int j = 0; j <= vd; j++) {
        atomicAdd(val + j, myValue[j]);
    }
}

template<int pd>
__global__ static void blur(int n, 
                            float *newValues,
                            const MatrixEntry3D *matrix,
                            const int *table_entries,
                            const signed short *table_keys,
                            const int table_capacity,
                            float *table_values,
                            int color,
                            const int vd)
{
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n) return;

    // Check if I'm valid
    if (matrix[idx].index != idx) return;

    // find my key and the keys of my neighbours
    short myKey[pd+1];
    short np[pd+1];
    short nm[pd+1];

    for (int i = 0; i < pd; i++) {
        myKey[i] = table_keys[idx*pd+i];
        np[i] = myKey[i]+1;
        nm[i] = myKey[i]-1;
    }


    np[color] -= pd+1;
    nm[color] += pd+1;

    int offNp = hashTableRetrieve<pd>(np, table_entries, table_keys, table_capacity);
    int offNm = hashTableRetrieve<pd>(nm, table_entries, table_keys, table_capacity);

    float *valMe = table_values + (vd+1)*idx;
    float *valNp = table_values + (vd+1)*offNp;
    float *valNm = table_values + (vd+1)*offNm;
    float *valOut = newValues + (vd+1)*idx;

    if (offNp >= 0 && offNm >= 0) {
        for (int i = 0; i <= vd; i++) {
            valOut[i] = (valNp[i] + (valMe[i]*2) + valNm[i])/2;
        }
    } else if (offNp >= 0) {
        for (int i = 0; i <= vd; i++) {
            valOut[i] = (valNp[i] + (valMe[i]*2))/2;
        }
    } else if (offNm >= 0) {
        for (int i = 0; i <= vd; i++) {
            valOut[i] = (valNm[i] + (valMe[i]*2))/2;
        }
    } else {
        for (int i = 0; i <= vd; i++) {
            valOut[i] = valMe[i];
        }
    }
}

template<int pd, typename Dtype>
__global__ static void slice(const int w, 
                             const int h,
                             const int d, 
                             const int vd,
                             Dtype *values,
                             const MatrixEntry3D *matrix,
                             float *table_values,
                             bool add) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;
    const int threadId = threadIdx.z * blockDim.x * blockDim.y
                          + threadIdx.y*blockDim.x 
                          + threadIdx.x;
    const int idx = z * h * w + y * w + x;
    const bool outOfBounds = (x >= w) || (y >= h) || (z >= d);

    if (outOfBounds) return;

    extern __shared__ float localValue[];

    float *myValue = localValue + threadId * vd;

    for (int i = 0; i < vd; i++) {
        myValue[i] = 0;
    }

    for (int i = 0; i <= pd; i++) {
        MatrixEntry3D r = matrix[idx * (pd + 1) + i];
        const float *val = table_values + r.index * (vd + 1);
        for (int j = 0; j < vd; j++) {
            myValue[j] += r.weight * val[j];
        }
    }

    float alpha = 1.0f / (1 + powf(2, -pd));
    for (int j = 0; j < vd; j++){
        if(!add){
            values[j * w * h * d + idx] = 0;
        }
        values[j * w * h * d + idx] += myValue[j] * alpha;
    }
}


template<int pd>
void gpu_init(const float* features,
              HashTable* table,
              MatrixEntry3D* matrix,
              const int w, 
              const int h,
              const int d)
{
    int num_points = w * h * d ;
    dim3 blocks((w - 1) / BLOCK_SIZE_XY + 1,
                (h - 1) / BLOCK_SIZE_XY + 1,
                (d - 1) / BLOCK_SIZE_Z + 1);
    dim3 blockSize(BLOCK_SIZE_XY, BLOCK_SIZE_XY, BLOCK_SIZE_Z);

    float blurVariance = 0.5;
    float * scaleFactor;
    float * scaleFactorHost = new float[pd];

    // Create Scale factor vector and give it to GPU
    // num_dimensions is likely to be low so do that
    // on the CPU
    for (int i = 0; i < pd; i++) {
        scaleFactorHost[i] = (pd+1)*sqrtf((1.0/6 + blurVariance)/((i+1)*(i+2)));
    }
    CUDA_CHECK(cudaMalloc((void**)&scaleFactor, sizeof(float)*pd));
    CUDA_CHECK(cudaMemcpy(scaleFactor, scaleFactorHost, sizeof(float)*pd, cudaMemcpyHostToDevice));

    createMatrix<pd><<<blocks, blockSize>>>(
            w, 
            h,
            d,
            features,
            table->table_entries,
            table->table_capacity,
            table->table_keys,
            scaleFactor,
            matrix);
    CUDA_POST_KERNEL_CHECK;
    // fix duplicate hash table entries
    dim3 cleanBlocks((num_points - 1) / CLEAN_BLOCK_SIZE + 1, 
                     2 * (pd + 1), 
                     1);
    cleanHashTable<pd><<<cleanBlocks, CLEAN_BLOCK_SIZE>>>(
        2 * num_points * (pd + 1),
        table->table_entries, 
        table->table_capacity, 
        table->table_keys);
    CUDA_POST_KERNEL_CHECK;

    blocks.z *= pd + 1;
    resetIndex<pd><<<blocks, blockSize>>>(
        w, 
        h, 
        d,
        matrix, 
        table->table_entries) ;
    CUDA_POST_KERNEL_CHECK;

    // Clean intermediate variables
    delete[] scaleFactorHost;
    CUDA_CHECK(cudaFree(scaleFactor));

}

template<int pd, typename Dtype>
void gpu_compute(Dtype* out, 
                 const Dtype* in, 
                 const HashTable &table,
                 const MatrixEntry3D* matrix,
                 int w, 
                 int h,
                 int d, 
                 int vd,
                 bool reverse, 
                 bool add){

    // Create table_values
    int num_points = w * h * d;
    float *table_values ;
    CUDA_CHECK(cudaMalloc((void**)&table_values, 
                          sizeof(float) * (vd + 1) * num_points * (pd + 1)));
    gpu_set<float>(num_points * (vd + 1) * (pd + 1), 0, table_values) ;

    dim3 blocks((w - 1) / BLOCK_SIZE_XY + 1,
                (h - 1) / BLOCK_SIZE_XY + 1,
                (d - 1) / BLOCK_SIZE_Z + 1);
    dim3 blockSize(BLOCK_SIZE_XY, BLOCK_SIZE_XY, BLOCK_SIZE_Z);

    // splat splits by color, so extend the z coordinate to our blocks to represent that
    blocks.z *= (pd + 1);
    splatCache<pd, Dtype><<<blocks, 
                            blockSize, 
                            BLOCK_SIZE * (vd + 1) * sizeof(float)>>>(
                                w, 
                                h, 
                                d,
                                vd,
                                in,
                                matrix,
                                table_values);
    CUDA_POST_KERNEL_CHECK;

    // blur
    dim3 cleanBlocks((num_points - 1) / CLEAN_BLOCK_SIZE + 1, 2 * (pd + 1), 1);
    float *newValues;
    float *oldValues;
    size_t size =  num_points * (pd + 1) * (vd + 1) * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&(newValues), size));
    CUDA_CHECK(cudaMalloc((void**)&(oldValues), size));
    gpu_set<float>(num_points * (vd + 1) * (pd + 1), 0, newValues) ;
    for (int color = reverse ? pd : 0; 
         color <= pd && color>=0; 
         reverse ? color-- : color++) {
        blur<pd><<<cleanBlocks, CLEAN_BLOCK_SIZE>>>(num_points * (pd + 1), newValues,
                matrix,
                table.table_entries,
                table.table_keys,
                table.table_capacity,
                table_values,
                color,
                vd);
        CUDA_POST_KERNEL_CHECK;
        // swap pointers does not seem to work...
        swapHashTableValues(oldValues, newValues, table_values, size);
    }

    // slice
    blocks.z /= (pd + 1);
    slice<pd, Dtype><<<blocks, 
                       blockSize, 
                       sizeof(float) * BLOCK_SIZE * vd>>>(
                            w, 
                            h, 
                            d,
                            vd, 
                            out, 
                            matrix, 
                            table_values, 
                            add);
    CUDA_POST_KERNEL_CHECK;

    // Free memory
    CUDA_CHECK(cudaFree(table_values)) ;
    CUDA_CHECK(cudaFree(newValues)) ;
    CUDA_CHECK(cudaFree(oldValues)) ;
}

void ModifiedPermutohedral3D::init_gpu(const float* features, 
                                     int num_dimensions, 
                                     int w, 
                                     int h,
                                     int d) {
    //Initialize Hash table
    if(!is_init){
        table.createHashTable(w * h * d * (num_dimensions+1), num_dimensions);
        int matrixSize = 
            sizeof(MatrixEntry3D) * (w * h * d * (num_dimensions+1));
        CUDA_CHECK(cudaMalloc((void **)&matrix, matrixSize));
    } else {
        table.resetHashTable(w_ * h_ * d_ * (ndim_ + 1), ndim_);
    }
    w_ = w;
    h_ = h;
    d_ = d;
    ndim_ = num_dimensions ;
    N_ = w * h * d ;
    switch(ndim_) {
      case 2:
        gpu_init<2>(features, &table, matrix, w_, h_, d_);
        break;
      case 3:
        gpu_init<3>(features, &table, matrix, w_, h_, d_);
        break;
      case 4:
        gpu_init<4>(features, &table, matrix, w_, h_, d_);
        break; 
      case 5:
        gpu_init<5>(features, &table, matrix, w_, h_, d_);
        break;
      case 6:
        gpu_init<6>(features, &table, matrix, w_, h_, d_);
        break;
      case 7:
        gpu_init<7>(features, &table, matrix, w_, h_, d_);
        break;    
      case 8:
        gpu_init<8>(features, &table, matrix, w_, h_, d_);
        break;
      case 9:
        gpu_init<9>(features, &table, matrix, w_, h_, d_);
        break;
      case 10:
        gpu_init<10>(features, &table, matrix, w_, h_, d_);
        break;  
      default:
        std::cout << "num_dimensions should be in [2, 10] but is " << ndim_ << " for init\n";
    }
    is_init = true;
}

void ModifiedPermutohedral3D::compute_gpu(float* out, 
                                        const float* in, 
                                        int value_size, 
                                        bool reverse, 
                                        bool add) const {
    // Losing time by dynamically allocating memory but more general function
    if(!is_init)
        std::cout << "Initialize lattice before doing any computing";
    switch(ndim_) {
      case 2:
        gpu_compute<2, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;
      case 3:
        gpu_compute<3, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;  
      case 4:
        gpu_compute<4, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;
      case 5:
        gpu_compute<5, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;
      case 6:
        gpu_compute<6, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;
      case 7:
        gpu_compute<7, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;  
      case 8:
        gpu_compute<8, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;
      case 9:
        gpu_compute<9, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;    
      case 10:
        gpu_compute<10, float>(out, in, table, matrix, w_, h_, d_, value_size, reverse, add);
        break;  
      default:
        std::cout << "num_dimensions should be in [2, 10] but is " << ndim_ << "\n";            
    }
}
