# Parallel-Ultrametric 
> This package implements a new matrix multiplication method that can transfer an arbitary dissimilarity matrix into an ultrametric distance matrix. Based on this method, this package also implements two GPU based functions as follows:
* data clusterability: Defined as the degree of the hardness of the transfer process. 
* Single-linkage hierarchical clustering: The final ultrametric distance matrix can generate a hierarchical structure on the dataset. Such structure is identical to that from the single-linkage hierarchical clustering.  

The whole package is coded in CUDA and C++ and embeded with PyTorch. 

You have to make sure your device is installed with the newest version of pytorch and CUDA. 


## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)
* [Inspiration](#inspiration)
* [Contact](#contact)

## General info
Add more general information about project. What the purpose of the project is? Motivation?

## Screenshots
![Example screenshot](./img/screenshot.png)

## Technologies
* Tech 1 - version 1.0
* Tech 2 - version 2.0
* Tech 3 - version 3.0

## Setup
Describe how to install / setup your local environement / add link to demo version.

## Code Examples
Show examples of usage:
`put-your-code-here`

## Features
List of features ready and TODOs for future development
* Awesome feature 1
* Awesome feature 2
* Awesome feature 3

To-do list:
* Wow improvement to be done 1
* Wow improvement to be done 2

## Status
Project is: _in progress_, _finished_, _no longer continue_ and why?

## Inspiration
Add here credits. Project inspired by..., based on...

## Contact
Created by [@flynerdpl](https://www.flyne

# Parallel-Ultrametric
This package calculates the ultrametric matrix in parallel 

The funciton is embeded with pytorch and coded in CUDA and C++. 
You have to make sure your device is installed with the newest version of pytorch and CUDA. 

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