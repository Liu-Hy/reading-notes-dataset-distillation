# Reading Notes on Dataset Distillation
## Factorization-based methods
### Dataset Condensation via Efficient Synthetic-Data Parameterization
**Motivation**: Make more efficient representation of data by considering data regularity; improve the optimization algorithm in previous gradient matching-based methods.

**Method**: 
- Use a differentiable augmentation method ("multi-formation") on the condensed data: partition each image into $2 \times 2$ regions and upsample them to the original resolution
- Propose a refined algorithm based on the gradient matching method in . Most importantly, they propose to train the model on the full training set instead of the synthetic dataset. They deem that the common practice of training the model $\theta_t$ on the condensed dataset $\mathcal{S}$ has two shorcomings: (1) the strong coupling of $\theta_t$ and $\mathcal{S}$ leads to a chicken-egg problem that makes the optimization unstable; (2) the small size of $\mathcal{S}$ leads to quick overfitting and vanishing gradient, making it hard to match the gradient using a distance-based loss. The benefit of training on the real dataset was shown through ablation studies. They also observed that models in the early training phase has gradients that are more informative for dataset distillation.

**Advantages**: 
- Both their proposed multi-formation method, and findings in optimization techniques are simple yet effective. The multi-formation is compatible with other strong augmentation methods such as Augmix to further improve performance.

**Limitation**: 
- Mostly a data augmentation method, their method utilizes the spatial redundancy of images, but does not really capture the relationship between data samples, so the compression ratio is not high. They also did experiment on multi-scale and learnable variants of multi-formation, but the results were not decisive.

### Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks
**Motivation**: Previous data distallation methods assign learnable examples for each class separately. However, there is information shared between classes that can be utilized.

**Method**: 
- Consider each synthetic image as a linear combination of a set of base vectors $\mathcal{M}=\\{\mathbf{b}_1, \dots, \mathbf{b}_K\\}$, where each $\mathbf{b}_k \in \mathbb{R}^d$ has the same dimension as $\mathbf{x}_k$. They learn a set of matrices $\\{\mathbf{A}_1, \dots, \mathbf{A}_r\\}$ where each $\mathbf{A}_i \in \mathbb{R}^{d_y \times K}$ maps the label $\mathbf{y} \in \mathbb{R}^{d_y \times 1}$ to coefficients of the bases. Then for $i=1, \dots, r$,
$${\mathbf{x}_i^{\prime}}^T = \mathbf{y}^{T} \mathbf{A}_i [\mathbf{b}_i; \dots ;\mathbf{b}_k]^T$$
- They base their work on the dataset distillation method in . They used the same BPTT algorithm to computate hypergradient, but found that using SGD with momentum and unrolling more steps is important to high performance. 

**Advantages**: 
- Captures the inter-example and inter-class relationships by sharing the bases as well as the mapping from label to coefficients. The parameter amount doesn't scale with the number of classes, which is beneficial in multi-class settings
- They showed that the learned compressed data can be used to synthetize new classifiers in two settings (Section 5.3).

**Limitations**: 
- Their inner training requires too many steps (e.g. 200), which is time-consuming. They use an offline method where the model is initialized from scratch after each update of the synthetic data, which may be inefficient. It is unclear how their method would work with an online method.
- They use fixed one-hot labels during training, while their formulation seems to have more potentials for soft labels and label embeddings.

### Dataset Distillation via Factorization
**Motivation**: capture the relationship between data examples for more efficient representation

**Method**: 
- Factorize the dataset into the cartisian product of a set of bases $$ and a set of hallucinators $$

