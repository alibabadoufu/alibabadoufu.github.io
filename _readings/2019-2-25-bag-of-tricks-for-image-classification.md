---
layout: single
title: "[Draft] Bag of Tricks for Image Classification with Convolutional Neural Networks"
date: 2019-2-25
tags: [deep learning, paper, amazon, convolutional neural network, tricks]
excerpt: "Deep Learning, Paper Reading, Convolutional Neural Network"
comments: true
toc: true
classes: wide
mathjax: "true"
excerpt: "This paper examined various refinements that have impacts on the final model accuracy"
header:
  teaser: "/images/readings/resnet.jpg"
  overlay_image: "/images/readings/resnet.jpg"
  overlay_filter: 0.6
  caption: "Photo credit: [**Bag of Tricks for Image Classification with Convolutional Neural Networks**](https://arxiv.org/abs/1812.01187)"
---

Refinements play an important role in deep learning workflow. These includes changes in loss functions, data preprocessing and optimization methods. These refinements improve the model accuracy by a few percentage (which is quite significant if you were to compare the state-of-the-art models in a specific category, for example image captioning) but they are typically not covered in details in the papers. In this paper, we can learn different tricks that could be applied to our own model and their impacts.

In the following sections, I will break down the discussion into different sections. These sections will either include discussion on individual chapter or a group of chapters.

## Introduction
Examine collection of training and model architecture refinements
- minor tricks, for example: modifying stride
- adjusting learning rate schedule
Together they make big difference

These tricks will be evaluated on multiple network and impact are reported to the final model accuracy

Advantage of their Tricks
- Can be generalize to other networks (inception and mobilenet) and other datasets (Place365)
- Bring better transfer learning performance in other applications such as object detection and semantic segmentation


## Training procedure
The template that the network uses is mini-batch stochastic gradient descent.
Author argued that functions and hyper-parameters in the baseline algo can be implemented in different ways

### Baseline training procedure
The preprocessing pipelines between training and validation are different

During training:
1. randomly sample an image and decode it into 32-bit raw values in [0,255]
2. crop a rectangular region whose aspect ratio is randomly sampled in [3/4,4/3] and area randomly sampled in [8%, 100%], then resize the cropped image into 224x224
3. Flip horizontally with 0.5 probability
4. Scale hue, saturation, and brightness with coefficients uniformly drawn from [0.6,1.4]
5. Add PCA noise with a coefficient sampled from a normal distribution $$N$$(0,0.1)
6. Normalize RGB channels by subtracting the pixels with 123.68, 116.779, 103.939 and dividing by 58.393, 57.12, 57.375

During validation:
1. Resize each image's shorted edge to 256 pixels while keeping its aspect ratio.
2. Crop a 224x224 region in the center
3. Normalize RGB channels the same way as step.no 6 for training data

Weights initialization:
1. Both convolutional and fully-connected layers are initialized with Xavier algorithm
2. In particular, parameters are set to random values uniformly drawn from [-a, a], where $$a = \sqrt{6/(d_{in}-d_{out})}$$. $$d_(in)$$ and $$d_(out)$$ are the input and output channel sizes, respectively.
3. All biases are initilized to 0
4. For batch normalization layers, $$\gamma$$ vectors are initilized to 1 and $$\beta$$ vectors to 0.

Optimizer:
1. Nesterov Accelerated Gradient (NAG) descent is used for training.
2. Each model trained for 120 epoch on 8 Nvidia V100 GPUs, batch_size set to 256.
3. Learning rate is initialized to 0.1 and divided by 10 at the 30th, 60th, and 90th epochs.
