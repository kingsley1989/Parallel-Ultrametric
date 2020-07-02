// ultMul_cuda.cpp
#include <torch/torch.h>

#include <vector>

// cuda ultMul declarations

at::Tensor ultMul_cuda
(
	at::Tensor A, 
	at::Tensor B
);

//cpp interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ultmul(torch::Tensor A, torch::Tensor B)
{
	CHECK_INPUT(A);
	CHECK_INPUT(B);
	
	return ultMul_cuda(A, B);
}

std::vector<torch::Tensor> clusterability(torch::Tensor D)
{
	CHECK_INPUT(D);

	torch::Tensor D_old = D.clone();
	torch::Tensor D_new = ultMul_cuda(D_old, D);
	int i = 1;

	while(!torch::all(torch::eq(D_old, D_new)).contiguous().item<bool>())
	{
		D_old = D_new.clone();
		D_new = ultMul_cuda(D_old, D);
		i++;
	}

	// i/|D| is the clusterability value and D_new is the sub-dominant ultrametric distance matrix of D.
	return {torch::tensor({i}), D_new};
}

torch::Tensor single_hclust(torch::Tensor D, int k)
{
	CHECK_INPUT(D);
	CHECK_INPUT(k);
	
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("ultmul", &ultmul, "ultmul (CUDA)");
	m.def("clusterability", &clusterability, "clusterability (CUDA)");
}


