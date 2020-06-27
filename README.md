# Parallel-Ultrametric 
> This package implements a new matrix multiplication method that can transfer an arbitary dissimilarity matrix into an ultrametric distance matrix. Based on this method, this package also implements two GPU based functions as follows:
* data clusterability: Defined as the degree of the hardness of the transfer process. 
* Single-linkage hierarchical clustering: The final ultrametric distance matrix can generate a hierarchical structure on the dataset. Such structure is identical to that from the single-linkage hierarchical clustering.  

The whole package is coded in CUDA and C++ and embeded with PyTorch. 

You have to make sure your device is installed with the newest version of PyTorch and CUDA. 


## Table of contents
* [General info](#general-info)
* [Prerequisite](#prerequisite)
* [Setup](#setup)
* [Demo Examples](#demo-examples)
* [Features](#features)
* [Future Work](#future-work)

## General info
Add more general information about project. What the purpose of the project is? Motivation?

## Prerequisite
* Python 3.5 and above
* CUDA 9 and above
* C++

## Setup
```python
import torch
import sklearn.metrics.pairwise as sk_dist
from torch.utils.cpp_extension import load

if torch.cuda.is_available():
    device = torch.device("cuda")
```

You can use JTL mode of this package as follows:

```python
ultMul = load(name='ultMul', sources=['ultMul_cuda_new.cpp', 'ultMul_cuda_kernel_new.cu'])

import ultMul as um
um.ultMul
```

## Demo Examples
demo example here speed comparison

## Features
List of features ready and TODOs for future development
* Awesome feature 1
* Awesome feature 2
* Awesome feature 3

To-do list:
* Wow improvement to be done 1
* Wow improvement to be done 2

## Future Work
Project is: _in progress_, _finished_, _no longer continue_ and why?



