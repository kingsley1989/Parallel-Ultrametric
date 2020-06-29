# Parallel-Ultrametric 
This package implements a new matrix multiplication method that can transfer an arbitary dissimilarity matrix into an ultrametric distance matrix. Based on this method, this package also implements two GPU based functions as follows:
> * Data Clusterability: Defined as the degree of the hardness of the transfer process. 
> * Parallel Hierarchical Clustering: The final ultrametric distance matrix can generate a hierarchical structure on the dataset. Such structure is identical to that from the single-linkage hierarchical clustering.  

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
Clustering is the prototypical unsupervised learning activity. It helps users to indentify cohesive and well-differentiated groups of records in data. In big data era, two main problems challenge users to apply clustering in practical scenario. The first is to determine whether a data set has non-random structure. The second is the fast clustering algorithm. 

Running clustering algorithm is expensive. If there are no meaningful well-differentiated groups in data, or say **not clusterable**, then it will be useless to execute clustering algorithm on such dataset. Developing a scale to determine whether a dataset is clusterable or say the **clusterability** of a dataset is an important issue. 

Due to the high time and space complexity of many clustering algorithms, especially for the linkage-based hierarchical clustering algorithms, the requirement of fast clustering is getting to be critical. Even if we can determine there's meaningful structure inside the dataset, we still hope to get such clustering result quickly. 

In this package, we developed a new matrix multiplication method that can transfer any dissimilarity matrix to a special ultrametic distance matrix. We adopt this techniques to deal with the problems above. 

In general, for a given dissimilarity matrix, this package can provide an extremely fast process from checking meaningful clustering structure to the final hierarchical clustering result.

### Ultrametric
The ultrametric is a special metric that has a stronger triangle inequality property:

![](https://latex.codecogs.com/gif.latex?\forall&space;x,y,z,d(x,y)\leq&space;max/{d(x,z),d(y,z)/})

Tansitive distance is a special ultrametric. It define the pairwise distance as the minimum hop (edge) of the set of largest edges along all possible connecting paths between two data points.

Definition
$TD(x,y) = min_{\mathbf{P}\in \mathbb{P}} max_{e\in \mathbf{P}} (d(e))$

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
The detailed demo examples can refer ult_test.ipynb for more information

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



