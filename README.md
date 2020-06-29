# Parallel-Ultrametric 
This package implements a new matrix multiplication method that can transfer an arbitary dissimilarity matrix into an ultrametric distance matrix. Based on this method, this package also implements two GPU based functions as follows:
> * Data Clusterability: Defined as the degree of the hardness of the transfer process. 
> * Parallel Hierarchical Clustering: The final ultrametric distance matrix can generate a hierarchical structure on the dataset. Such structure is identical to that from the single-linkage hierarchical clustering.  

The whole package is coded in CUDA and C++ and embeded with PyTorch. 

You have to make sure your device is installed with the newest version of PyTorch and CUDA. 


## Table of contents
* [Prerequisite](#prerequisite)
* [Setup](#setup)
* [Demo Examples](#demo-examples)
* [Background info](#background-info)
* [Reference](#reference)
* [Future Work](#future-work)

## Prerequisite
* Python 3.5 and above
* CUDA 9 and above
* PyTorch 1.0 and above

## Setup
```python
import torch
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

## Background Info
Clustering is the prototypical unsupervised learning activity. It helps users to indentify cohesive and well-differentiated groups of records in data. In big data era, two main problems challenge users to apply clustering in practical scenario. The first is to determine whether a data set has non-random structure. The second is the fast clustering algorithm. 

Running clustering algorithm is expensive. If there are no meaningful well-differentiated groups in data, or say **not clusterable**, then it will be useless to execute clustering algorithm on such dataset. Developing a scale to determine whether a dataset is clusterable or say the **clusterability** of a dataset is an important issue. 

Due to the high time and space complexity of many clustering algorithms, especially for the linkage-based hierarchical clustering algorithms, the requirement of fast clustering is getting to be critical. Even if we can determine there's meaningful structure inside the dataset, we still hope to get such clustering result quickly. 

In this package, we developed a new matrix multiplication method that can transfer any dissimilarity matrix to a special ultrametic distance matrix. We adopt this techniques to deal with the problems above. 

In general, for a given dissimilarity matrix, this package can provide an extremely fast process from checking meaningful clustering structure to the final hierarchical clustering result.

### Ultrametric
Detailed introduction of ultrametric could refer to [1] and [2].

The ultrametric is a special metric that has a stronger triangle inequality property:

$$\forall x,y,z\ d(x,y)\leq max\lbrace d(y,z), d(x,z)\rbrace$$
<!-- ![](https://latex.codecogs.com/png.latex?\forall&space;x,y,z,d(x,y)\leq&space;max\{d(x,z),d(y,z)\}) -->

Tansitive distance is a special ultrametric. It defines the pairwise distance as the minimum hop (edge) of the set of largest edges along all possible connecting paths between two data points.

***Definition 1*** 

> $$TD(x,y) = \min_{\mathcal{P}\in \mathbb{P}} \max_{e\in \mathcal{P}} (d(e))$$

Here $\mathcal{P}$ is a single path from point x to y, while $\mathbb{P}$ represents the set of all possible paths from point x to y.

### A Special Matrix Product
This and the next two sections could refer to [3] for detailed prove and illustration.

Let $\mathbb{P}_\infty = \lbrace x\in\mathbb{R}|x\geq 0\rbrace \cup\lbrace\infty\rbrace$.

Suppose $A\in \mathbb{P}_{\infty}^{m\times n}$ and 

$B\in \mathbb{P}_{\infty}^{n\times l}$, we have:

***Definition 2***

>$C = A\otimes B\in\mathbb{P}_\infty^{m\times l}$ such that,
>
>$c_{ij} = \min\lbrace \max\lbrace a_{ik}, b_{kj}\rbrace |1\leq k\leq n\rbrace$

Let $A\preceq B$ if $a_{ij}\geq b_{ij}$

***Theorem 1***
 
> If $A\in\mathbb{P}^{n\times n}$ is a dissimilarity matrix, there exists $m\in\mathbb{N}$ such that
>
> $A\preceq A^2\preceq\cdots\preceq A^m = A^{m+1}=\cdots = A^{m+d}, \forall d>0$
>
> and $A^m$ is an *ultrametric matrix*

### Ultrametricity and Clusterability
***Definition 3***

> Let $A\in\mathbb{P}^{n\times n}$ be the dissimilarity matrix of dataset $S$ and $m(A)$ is the least integer that $A^m$ is the ultrametric matrix, then the **ultrametricity** of $A$ $u(A)=\frac{n}{m}$

We refer to $m(A)$ as the *stabilization power* of the matrix $A$.

If $m(A)=1$, $A$ is ultrametric itself and $u(A)=n$.

***Definition 4***

> Let $A_D$ as the dissimilarity matrix of a data set $D$.
> 
> The **clusterability** of a data set $D$
> 
> $\mathtt{clust}(D) = u(A_D) = \frac{n}{m(A_D)}$

### Subdominant Ultrametric and Single-link Hierarchical Clustering
***Theorem 2***

> Let $(D, d)$ as the dissimilarity space, $A$ is the dissimilarity matrix, and $m$ is the stabilization power. 
> 
> Then, the new dissimilarity $d'$ with the dissimilarity matrix $A^m$ is the **subdominant ultrametric** for $d$.

From [4, 5, 6, 7], we can conclude that the cophenetic distance matrix $C$ of single-linkage hierarchical clustering on data set $D$ is identical to its subdominant ultrametric distance matrix $A^m$.
<!-- 4 for subdominant definition on min-max conn to transtive dist-->
<!-- 5 connect mst to single-link-->
<!-- 6 connect trans dist to mst-->
<!-- 7 is the newest work on ultra-->

## Reference
> [1] Rammal, R., Toulouse, G., & Virasoro, M. A. (1986). Ultrametricity for physicists. Reviews of Modern Physics, 58(3), 765.

> [2] Simovici, D. A., Vetro, R., & Hua, K. (2017). Ultrametricity of dissimilarity spaces and its significance for data mining. In Advances in Knowledge Discovery and Management (pp. 141-155). Springer, Cham.

> [3] Simovici, D., & Hua, K. (2019, October). Data ultrametricity and clusterability. In Journal of Physics: Conference Series (Vol. 1334, No. 1, p. 012002). IOP Publishing.

> [4] Bayod, J. M., & Martinez-Maurica, J. (1990). Subdominant ultrametrics. Proceedings of the American Mathematical Society, 109(3), 829-834.

> [5] Gower, J. C., & Ross, G. J. (1969). Minimum spanning trees and single linkage cluster analysis. Journal of the Royal Statistical Society: Series C (Applied Statistics), 18(1), 54-64.

> [6] Yu, Z., Xu, C., Meng, D., Hui, Z., Xiao, F., Liu, W., & Liu, J. (2014). Transitive distance clustering with k-means duality. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 987-994).

> [7] Chierchia, G., & Perret, B. (2019). Ultrametric fitting by gradient descent. In Advances in neural information processing systems (pp. 3181-3192).
## Future Work
This work only fits for one GPU calculation. The next step should focus on multi-GPU work. The multi-GPU calculation can demonstrate the algorithm's best advantage.



