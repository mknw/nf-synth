# NF Synth

*The present code is in prototype stage.*

This repository contains the code for the MSc Thesis: *"Normalising Flow: Visualising Latent Space"*. Written and coded by Michael Accetto and supervised by Jakub Tomczak.

## Project

It is intended to compress images with continuous, Normalising Flow (NF) generative models.

The NF architectures are: 

- Glow ([link to paper](https://arxiv.org/abs/1807.03039))
- ~~Real-NVP~~ ([link to paper](https://arxiv.org/abs/1605.08803))

## Usage 

Network training:

	$ python3 train.py [--config /path/to/config.yml]

Compression visualisation:

	$ python3 reduce.py [--config /path/to/config.yml]

Similarity analysis:

	$ python3 similarity.py [--config /path/to/config.yml]

## Models

### Synthesizer

Abstraction over step based compression module comprising:

- Normalising Flow Architecture (Glow or ~Real NVP~)
- Principal Component Analysis (PCA)
- Uniform Manifold Approximation Projection (UMAP, optional)

`Synthesizer` is a scikit compliant transformer class implementing the methods: `fit`, `transform`,
`inverse_transform`, `fit_transform`.

A more detailed description will be soon pushed to `main`.


## Architectures

### Glow

Trainable on CelebA-128 and FFHQ-128.
Additionally, the model contains a learned prior at the end of each flow-step. 

Glow samples, FFHQ dataset:
![Samples on training FFHQ-128](n-16_sample_t-0.50.png)
