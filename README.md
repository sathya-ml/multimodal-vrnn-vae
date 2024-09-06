# Multimodal Variational Recurrent Neural Network and Multimodal Variational Autoencoder Implementation

## Introduction

This project contains a PyTorch implementation of a Multimodal Variational Recurrent Neural Network (MM-VRNN) and a Multimodal Variational Autoencoder (MM-VAE). The same were used to produce part of the results for the PhD Thesis ["On Wiring Emotion to Words: A Bayesian Model", 2022](https://air.unimi.it/bitstream/2434/932589/2/phd_unimi_R12197.pdf) that included a multimodal generative approach to modelling affect dynamics.

The *Variational Autoencoder (VAE)* is a generative model first introduced by [Kingma et al. (2013)](https://arxiv.org/abs/1312.6114). To learn more about the VAE, aside from the original paper I would recommend [this blog post](https://lilianweng.github.io/posts/2018-08-12-vae/). The *Variational Recurrent Neural Network (VRNN)* by [Chung et al. (2015)](https://arxiv.org/abs/1511.06349) is instead a development over the standard recurrent neural network (RNN) that includes a VAE to learn a latent representation of the input data at each time step. The VRNN, containing the core of a RNN, allows for improved modelling of temporal dynamics in sequential data w.r.t. both standard RNNs and standard VAEs.

This repository focuses on rendering both these models multimodal, so that they can be used to model a unique latent representation of different data modalities, both in the case of sequential data and non-sequential data. Multimodal data is ubiquitous in nature, and being able to model it is a key feature for many areas of application of ML models, like affective computing, human-computer interaction and the creative industry.

There's more than one approach to making a ML model multimodal, and thus there are other implementations of multimodal VAEs and VRNNs, of which the most relevant are listed in the [References](#references) section. This project takes the following two approaches, respectively:

+ For the MM-VAE, each modality get its own encoder and decoder. During enconding, the latent space is shared and the inferences of each modality are joined through an expert layer. In the decoding process, each modality's decoder receives the latent variable sample as input and produces a reconstruction of the original data. The only expert implemented here is a non-learnable Product of Experts. This approach is very similar to the approach of [Wu and Goodman, 2018](https://arxiv.org/pdf/1802.05335).

+ For the MM-VRNN, the approach is different. Here, the original model is kept the same and an additional layer is added for both the encoding and decoding process. In the inference stage each modality gets its own encoder and the encodings are fused by simple vector concatenation. In the case of missing modalities, a zero tensor of appropriate dimensions is passed forward. This then gets passed to the following layer, which is the input layer of the original VRNN. In the decoding process, the latent variable is sampled and passed to each modality's decoder, which then produces a reconstruction of the original data.

Both of the above approaches effectively allow for arbitrary conditioning and missing modalities, which are extremely important for any real-world application of a multimodal ML model. The expert approach could be relatively easily improved by learning the expert's weights from data. Instead, the concatenation approach is more rudimentary, and using a predefined value for the missing modalities is not optimal. However, it was chosen for its simplicity, flexiblity and because it doesn't require modifying significantly the original model architecture, which is quite complex as it is.

Emphasis was placed on code clarity and modularity, and therefore there is some redundancy in the code. But this allows for easy understanding of the model and its components, and modifying the code for future developments. Furthermore, both models have been tried and tested in the context of multimodal affective computing and produce good results, but require some hyperparameter tuning to get the best results.

## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Features and Usage

The primary contribution of this project are the MM-VAE and MM-VRNN modules contained in [src/model](src/model/). This folder further contains the `Expert` abstract class with the appropriate method signature for forward, along with a Product of Experts implementation. The modules are intended to be modular and extensible. It takes in input other modules for each modality, which you should construct yourself and pass as arguments to the constructor. This offers flexibility regarding modalities and networks that process them. You could easily use an MLP, CNN, or any other network architecture you deem useful for your problem.

In addition, provided in [src](src/) are two toy examples of how to train the models, one for the MM-VAE and another for the MM-VRNN, respectively. In the sections below you can find more details on particular aspects of the implementation. To run the examples run:

```bash
python mmvae_example.py
```

for the MM-VAE example, and for the MM-VRNN example run:

```bash
python mmvrnn_example.py
```

### The Datasets

To test the models two simple synthetic datasets are provided. The code for generating the datasets can be found in [dataset.py](src/dataset.py). The Spirals Dataset generates two interdependent 2D modalities. For each sample, the first modality creates a single 2D spiral using parametric equations with sine and cosine functions. The spiral's amplitude is randomly varied. The second modality is derived from the first through a linear transformation of its x and y coordinates. Both modalities are then perturbed with Gaussian noise. This dataset simulates temporally dependent, multi-modal data with a clear relationship between modalities. Given that it has a temporal component, it is the most suitable dataset to test the MM-VRNN.

The Quadratic Sinusoidal Dataset creates two interrelated 2D modalities. The first modality combines a quadratic function $y = x^2$ for $y_1$ with a linear distribution for $x_1$. The second modality is derived from the first: $x_2$ is based on $y_1$, while $y_2$ combines a sinusoidal function of $x_1$ with a scaled $y_1$. Both modalities incorporate Gaussian noise to simulate real-world data variability. This dataset is used for testing the MM-VAE.

### Importance of $\beta$, the KL-Divergence Loss Weight

In the equation of the loss function for the VAE, the KL-Divergence term may be multiplied by a hyperparameter $\beta$. This hyperparameter can be used to control the balance between the reconstruction loss and the KL-Divergence loss. When $\beta$ is small (close to 0) the model is only reconstructing the data without considering much the KL-Divergence loss, which makes it equivalent to a plain autoencoder. Instead, when $\beta$ is large (>1), the model is only considering the KL-Divergence loss which has the effect of making the latent space prior closer to a standard Gaussian distribution and may improve the learning of disentangled latent representations ([see Higgins et al. (2017)](https://openreview.net/pdf?id=Sy2fzU9gl)). On the other hand, when $\beta$ is too large, it may exacerbate the problem of posterior collapse, where the model collapses to a standard Gaussian distribution, losing the ability to learn a good latent representation of the data.

In practice, $\beta$ can be used to control the trade-off between generating realistic data and ensuring the model is learning a good latent representation of the data. It is not always straightforward to choose the best value for $\beta$, and bear in mind that this is a very important hyperparameter for the VAE family of models and can strongly affect training outcomes.

There are reasons to believe, as per the work of [Fu et al. (2019)](https://arxiv.org/pdf/1903.10145) that a cyclical annealing schedule for $\beta$ may improve performance and stability of training and help avoid the vanishing KL-Divergence problem. Included in this repository in [annealing.py](src/model/annealing.py) is a class that can be used to perform such annealing, but it is not used in the examples provided. It contains two functions that return generators that generate a $\beta$ according to a linear, or a cyclical linear annealing schedule, respectively.

### Modality Dropout

At least in the case of the MM-VRNN, the network has to somehow learn how a missing modality is represented due to the way the model is structured. This constrains us to include missing modalities in the training phase. Similar reasoning also stands in the case of the MM-VAE, and it is beneficial to include missing modalities in the during training. For $N$ modalities, there are $2^N - 2$ viable possible combinations of missing modalities, so using them all systematically would result in a very large number of training cycles. An alternative approach is to use modality dropout, a technique where, during training, one or more modalities are randomly removed from the input data. This has the benefit of improving robustness and preventing over-reliance on any single modality.

Modality dropout was first introduced in [Neverova et al. (2015)](https://arxiv.org/pdf/1501.00102). In this project, however, a very rudimentary form of it is implemented where a dropout probability is set, and if triggered, one or more modalities (sampled uniformly) are randomly removed from the input data at each iteration.

## References

**Other multimodal VRNN implementations** (the list is not exhaustive, and the order is irrelevant):

+ [Multimodal Deep Markov Model (MDMM)](https://github.com/ztangent/multimodal-dmm)
+ [A Multimodal Predictive Agent Model for Human Interaction Generation](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w66/Baruah_A_Multimodal_Predictive_Agent_Model_for_Human_Interaction_Generation_CVPRW_2020_paper.pdf)
+ [Social-VRNN: One-Shot Multi-modal Trajectory Prediction for Interacting Pedestrians](https://proceedings.mlr.press/v155/brito21a/brito21a.pdf), [[Code]](https://github.com/tud-amr/social_vrnn)
+ [Modeling Emotion in Complex Stories: The Stanford Emotional Narratives Dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8913483&casa_token=EQfQ5eQNEgAAAAAA:lvE9NLGsecEoBOGQrmGKn7ab0b3LpJ75xDbyrsXpYUL4W1c6fzE-t-EVLnMPDscNu5kBKjhR&tag=1), [[Code]](https://github.com/desmond-ong/TAC-EA-model)

For other Multimodal VAE implementations see [here](https://arxiv.org/pdf/2209.03048) a list and benchmark comparisons.

## License

This project is licensed under the MIT License. See the [LICENSE file](LICENSE) for more details.
