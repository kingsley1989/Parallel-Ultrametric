// ultMul__cuda_kernel.cu 
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits.h>
#include <algorithm>
/*
// get element of matrix A at [row, col]
template <typename scalar_t>
__device__ __forceinline__ scalar_t getElement(scalar_t* M, unsigned int row, unsigned int col, size_t width)
{
	return M[row * width + col];
}


// assign element of matrix A at [row, col]
template <typename scalar_t>
__device__ __forceinline__ void setElement(scalar_t* M, unsigned int row, unsigned int col, size_t width, scalar_t v)
{
	M[row * width + col] = v;
}
*/

// kernel function for ultrametric matrix multiplication
template <typename scalar_t>
__global__ void ultMul_cuda_kernel(
	const scalar_t* __restrict__ A, 
	const scalar_t* __restrict__ B, 
	scalar_t* __restrict__ C, size_t k, size_t m, size_t n) // k = A.width, n=B.width
{
    scalar_t elmt = INT_MAX-1.0;
    scalar_t temp = 0.0;
    
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) //only vlaue within the matrix should be calculated and assigned
    {
        for (int i = 0; i < k; ++i)
        {
            //temp = fmaxf(getElement(A, row, i, k), getElement(B, i, col, n));
            temp = fmaxf(A[row * k + i], B[i * n + col]);
            elmt = fminf(elmt, temp);
        }
        //setElement(C, row, col, n, elmt);
        C[row * n + col] = elmt;
    }
}


at::Tensor ultMul_cuda(at::Tensor A, at::Tensor B)
{
    const auto m = A.size(0); // tensor row m*k
    const auto k = A.size(1);
    const auto n = B.size(1); // tensor column k*n

    auto C = torch::zeros({m,n}, torch::CUDA(at::kFloat));//intArrayRef type for size of tensor
    
    dim3 threads(n, m); //blocksize
    dim3 blocks(1, 1); //gridsize
    if (m*n > 512)
    {
        threads.x = 16;
        threads.y = 16;
        blocks.x = ceil(float(n)/float(threads.x));
        blocks.y = ceil(float(m)/float(threads.y));
    }
    
    cudaDeviceSynchronize();
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ultMul_cuda", ([&] {
        ultMul_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            k, m, n);
    }));

    return C;
}

