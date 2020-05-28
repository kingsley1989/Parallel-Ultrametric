from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name='ultMul', 
	ext_modules=[
	CUDAExtension(
		'ultMul_cuda',
[
			'ultMul_cuda_new.cpp',
			'ultMul_cuda_kernel_new.cu', #.cpp and .cu file must have different name
		])],
	cmdclass = {
		'build_ext': BuildExtension
	}
)
