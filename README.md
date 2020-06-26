# Parallel-Ultrametric
This package calculates the ultrametric matrix in parallel 

The funciton is embeded with pytorch and coded in CUDA and C++. 
You have to make sure your device is installed with the newest version of pytorch and CUDA. 

``python
import torch
import sklearn.metrics.pairwise as sk_dist
from torch.utils.cpp_extension import load

if torch.cuda.is_available():
    device = torch.device("cuda")
``

You can use JTL mode of this package as follows:

ultMul = load(name='ultMul', sources=['ultMul_cuda_new.cpp', 'ultMul_cuda_kernel_new.cu'])

import ultMul as um
um.ultMul