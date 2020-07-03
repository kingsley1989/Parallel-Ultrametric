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

std::vector<torch::Tensor> single_hclust(torch::Tensor D, int k)
{
	CHECK_INPUT(D);
	
	torch::Tensor D_old = D.clone();
	torch::Tensor D_new = ultMul_cuda(D_old, D);

	while(!torch::all(torch::eq(D_old, D_new)).contiguous().item<bool>())
	{
		D_old = D_new.clone();
		D_new = ultMul_cuda(D_old, D);
	}
   //output: first is the sorted result(long list<=D_new.size(0) )and second is the index
	std::tuple<torch::Tensor, torch::Tensor> X = torch::_unique(D_new, true, true);
   torch::Tensor x1 = std::get<0>(X);
   torch::Tensor Dsub = torch::nonzero(D_new <= x1[-k]).to(torch::kInt));
	return {x1, Dsub};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("ultmul", &ultmul, "ultmul (CUDA)");
	m.def("clusterability", &clusterability, "clusterability (CUDA)");
	m.def("single_hclust", &single_hclust, "single_hclust (CUDA)");
}


