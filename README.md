# Reading Notes on Dataset Distillation
## Factorization-based methods
### Dataset Condensation via Efficient Synthetic-Data Parameterization
**Motivation**: Make more efficient representation of data by considering data regularity; improve the optimization algorithm in previous gradient matching-based methods.

**Method**: 
- Use a differentiable augmentation method ("multi-formation") on the condensed data: partition each image into $2 \times 2$ regions and upsample them to the original resolution
- Proposed a refined algorithm based on the gradient matching method in Zhao et al. [2]. Their main proposal is to train the model on the full training set instead of the synthetic dataset. They find that the common practice of training the model $\theta_t$ on the condensed dataset $\mathcal{S}$ has two shorcomings: (1) the strong coupling of $\theta_t$ and $\mathcal{S}$ leads to a chicken-egg problem that makes the optimization unstable; (2) the small size of $\mathcal{S}$ leads to quick overfitting and vanishing gradient, making it hard to match the gradient using a distance-based loss. The benefit of training on the real dataset was shown through ablation studies. They also observed that models in the early training phase has gradients that are more informative for dataset distillation.

**Advantages**: 
- Both their proposed multi-formation method, and findings in optimization techniques are simple yet effective. The multi-formation is compatible with other strong augmentation methods such as Augmix to further improve performance.

**Limitation**: 
- Mostly a data augmentation method, their method utilizes the spatial redundancy of images, but does not really capture the relationship between data samples, so the compression ratio is not high. They also did experiment on multi-scale and learnable variants of multi-formation, but the results were not decisive.

### Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks
**Motivation**: Previous data distallation methods assign learnable examples for each class separately. However, there is information shared between classes that can be utilized.

**Method**: 
- Consider each synthetic image as a linear combination of a set of base vectors $\mathcal{M}=\\{\mathbf{b}_1, \dots, \mathbf{b}_K\\}$, where each $\mathbf{b}_k \in \mathbb{R}^d$ has the same dimension as $\mathbf{x}_k$. They learn a set of matrices $\\{\mathbf{A}_1, \dots, \mathbf{A}_r\\}$ where each $\mathbf{A}_i \in \mathbb{R}^{d_y \times K}$ maps the label $\mathbf{y} \in \mathbb{R}^{d_y \times 1}$ to coefficients of the bases. Then for $i=1, \dots, r$,
$${\mathbf{x}_i^{\prime}}^T = \mathbf{y}^{T} \mathbf{A}_i [\mathbf{b}_i; \dots ;\mathbf{b}_k]^T$$
- They base their work on the dataset distillation method in Wang et al. [1]. They used the same BPTT algorithm to computate hypergradient, but found that using SGD with momentum and unrolling more steps is important to high performance. 

**Advantages**: 
- Captures the inter-example and inter-class relationships by sharing the bases as well as the mapping from label to coefficients. The parameter amount doesn't scale with the number of classes, which is beneficial in multi-class settings
- They showed that the learned compressed data can be used to synthetize new classifiers in two settings (Section 5.3).

**Limitations**: 
- Their inner training requires too many steps (e.g. 200), which is time-consuming. They use an offline method where the model is initialized from scratch after each update of the synthetic data, which may be inefficient. It is unclear how their method would work with an online method.
- They only use fixed one-hot labels during training, while their formulation seems to have more potentials for soft labels and label embeddings.

### Dataset Distillation via Factorization
**Motivation**: capture the relationship between data examples for more efficient representation

**Method**: 
- Factorize the dataset into the cartesian product of a set of bases and a set of hallucinators $\\{H_{\theta_j}\\}$, where the hallucinators are style transfer networks that scales and shifts the latent features of the bases. This results in $|\mathcal{H}| |\mathcal{B}|$ synthetic examples. Visualizationg of the learned representation shows that the bases mainly store the structure and contour information, while the hallucinators render the styles and details of the image.
- To encourage diversity of the hallucinators, trained a feature extractor and the hallucinators in an adversarial setting ("adversary contrastive constraint"). 
  - The feature extractor acts as an adversary that minimizes the divergence between the outputs of two different hallucinators from the same basis. Let $F$ denote the feature extractor and $F_{-1}$ denote the feature at the last hidden layer. $F$ is trained with a contrastive loss $\mathcal{L}_{con}$:
<p align="center">
  <img src="https://github.com/Liu-Hy/reading-notes-dataset-distillation/blob/main/imgs/HaBa%20eq3.png" width="500" height="72"/>
</p>

  - $F$ is also trained with $\mathcal{L}_{task}$, the cross-entropy loss in the classification task over the synthetic data:
<p align="center">
  <img src="https://github.com/Liu-Hy/reading-notes-dataset-distillation/blob/main/imgs/HaBa%20eq%204.png" width="280" height="36"/>
</p>

  - Meanwhile, the synthetic set $\mathcal{S}$ maximizes the divergence to increase diversity. Different from $\mathcal{L_{con}}$, they use cosine similarity as the metric:
<p align="center">
  <img src="https://github.com/Liu-Hy/reading-notes-dataset-distillation/blob/main/imgs/HaBa%20eq%205.png" width="400" height="72"/>
</p>

- $\mathcal{S}$ is also trained on $\mathcal{L}_{DD}$, the loss for dataset distillation:

<p align="center">
  <img src="https://github.com/Liu-Hy/reading-notes-dataset-distillation/blob/main/imgs/HaBa%20eq6.png" width="240" height="36"/>
</p>

**Advantages**: 
- Their method is generalizable. They applied their factorization to various dataset distillation methods and achieved consistent performance gain. 
- Using the same number of final images (and much less parameters), their method still outperforms the baseline consistently, which indicates that the inductive bias introduced by their method is beneficial (probably because the hallucinators capture the different styles that exist in the dataset).

**Limitation**: 
- In their method, the bases remain the same spatial resolution as the original images, while their formulation should allow for a much higher compression ratio. They experimented with reducing the channels of the bases to 1 and the performance did not reduce much. It remains to be seen whether they can reduce the spatial resolution as well.

### Dataset Condensation via Efficient Synthetic-Data Parameterization
**Motivation**: To better utilize the regularity in a given dataset, assume the data are generated from low-dimensional latent codes with neural decoders.

**Method**: 
- Factorize the dataset into the cartesian product of a set of low-dimensional latent codes and a set of decoders, where the decoders are generator networks with transpose convolution layers.
- They base their method on distribution matching in Zhao et al. [3], which uses randomly generated feature extractors $g(\dot)$ without training. Denote the set of real examples for a class $c$ as $\mathcal{X_c} = \\{x_{c,1}, \dots, x_{c,N}\\}$, $c=1, \dots, N$. Denote the set of latent codes for class $c$ as $\mathcal{\Theta_c} = \\{\theta_{c,1}, \dots, \theta_{c,M}\\}$. $f(\theta; \phi_d)$ is the dth decoder parameterized by $\phi_d$, and let $\phi = \\{\phi_1, \dots, \phi_d\\}$. They use MMD loss to match the distribution:

<p align="center">
  <img src="https://github.com/Liu-Hy/reading-notes-dataset-distillation/blob/main/imgs/KFS%20eq1.png" width="500" height="66"/>
</p>

- They use full batch training for both real and synthetic data. They observe that previous methods that subsample synthetic data in mini-batches produce biased gradients which hinder the diversity of synthetic examples, and that subsampling of real examples produces high-variance gradients degrading the quality of distillation. They also derived the formulas of gradient bias and variance in simplified settings.

**Advantages**: 
- As of January 2023, their method has the best overall performance in all data distillation methods. The good performance is consistent when limiting the number of training steps during evaluation, showing the effectiveness of their method.
- With a low-dimensional latent code and a light-weight decoder network, their method has the highest compression ratio.
- Without a side component to encourage diversity, their method is able to generate diverse latent codes that capture the multi-modality of the data distribution.

**Limitations**:
- Full batch training is expensive, so they only experimented with distribution matching-based method to avoid computing hyper-gradients. More work needs to be done to make the algorithm more efficient and adapt to other data distillation algorithms
- It may be better to condition the decoders on label information because different labels may have a different set of modes in the dataset. 


## References
[1] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A. Efros. Dataset Distillation, Feb. 2020. arXiv:1811.10959 [cs, stat].\
[2] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset Condensation with Gradient Matching, Mar. 2021. arXiv:2006.05929 [cs]. \
[3] Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. arxiv:2110.04181 [cs].
