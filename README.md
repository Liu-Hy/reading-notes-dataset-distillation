# Reading Notes on Dataset Distillation
## Factorization-based methods
### Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks
Problem: \
Method: Generate each synthetic images as a linear combination of a set of bases: 
$${\mathbf{x}_i^{\prime}}^T = \mathbf{y}^{T} \mathbf{A}_i [\mathbf{b}_i; \dots ;\mathbf{b}_k]^T$$
