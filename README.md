# Reading Notes on Dataset Distillation
## Factorization-based methods
### Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks
**Aim**: Reduce data redundancy and improve performance of the distilled data \
**Method**: 
- Consider each synthetic image as a linear combination of a set of base vectors $\mathcal{M}=\\{\mathbf{b}_1, \dots, \mathbf{b}_K\\}$, where each $\mathbf{b}_k \in \mathbb{R}^d$ has the same dimension as $\mathbf{x}_k$. They learn a set of matrices $\\{\mathbf{A}_1, \dots, \mathbf{A}_r\\}$ where each $\mathbf{A}_i \in \mathbb{R}^{d_y \times K}$ maps the label $\mathbf{y} \in \mathbb{R}^{d_y \times 1}$ to coefficients of the bases. Then for $i=1, \dots, r$,
$${\mathbf{x}_i^{\prime}}^T = \mathbf{y}^{T} \mathbf{A}_i [\mathbf{b}_i; \dots ;\mathbf{b}_k]^T$$
- They base their work on the dataset distillation method in . They used the same BPTT algorithm to computate hypergradient, but found that using SGD with momentum and unrolling more steps is important to high performance. 

**Advantages**: 
- Captures the inter-example and inter-class relationships by sharing the base vectors and the mapping from label to coefficients. The parameter amount doesn't scale with the number of classes, which is beneficial in multi-class settings
- They showed that the learned compressed data can be used to synthetize new classifiers in two settings (Section 5.3).

**Limitations**: 
- Their inner training requires too many steps (e.g. 200), which is time-consuming. They use an offline method where the model is initialized from scratch after each update of the synthetic data, which may be inefficient. It is unclear how their method would work with an online method.
- They use fixed one-hot labels during training, while their formulation seems to have more potentials for soft labels and label embeddings.

