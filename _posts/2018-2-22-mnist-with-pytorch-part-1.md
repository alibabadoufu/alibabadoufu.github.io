---
layout: single
title: "MNIST Classifier with Pytorch [Part I]"
date: 2019-2-22
tags: [deep learning, pytorch, neural network]
excerpt: "Deep Learning, Programming, Pytorch"
comments: true
toc: true
---

I am doing a revision on how to build neural network with PyTorch. To do so I am taking Udacity's online lesson on [Intro to Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188). This course introduces many important models such as CNN and RNN using PyTorch. In the subsequent posts, I will try to summarize important points that I have learnt in the course.

## Prerequisite
For those who have learnt fundamental deep learning concepts should not find any difficulties to follow. However, if you are a fresh new beginner in this field, I would strongly encourage you to go through Andrew Ng's Deep Learning Specialization on Coursera before reading this entire post series.

## What is MNIST Dataset?
MNIST consists of greyscale handwritten digits ranging from 0 to 9. Each image is 28 x 28 pixels.

{% include figure image_path="https://corochann.com/wp-content/uploads/2017/02/mnist_plot.png" alt="This is MNIST Data" caption="MNIST" %}

## What is PyTorch?
As its name implies, PyTorch is a Python-based scientific computing package. It allows developers to compute high-dimensional data using tensor with strong GPU acceleration support. One of the advantages over Tensorflow is PyTorch avoids static graphs. This allows developers to change the network behavior on the fly.

I was reluctant to use PyTorch when I first started learning deep learning is because of it poor production support. However, recent release of PyTorch 1.0 has overcome the challenges. The merge between PyTorch and Caffe2 allows researchers to move seemlessly from research to production without worries about migration issue.

Overall speaking, it's always good to learn both Tensorflow and PyTorch as these two frameworks are designed by the two giant companies which focus heavily on Deep Learning development. There is no reason to choose either side especially for someone who wishes to make their models reachable to the community.

## Basic Workflow
In this section, we will discuss about the basic workflow of classifying image using PyTorch.

### Import Library
To build the model, we need the tools. We first import the libraries which are needed for our model.

```python
    import torch
    from torch import nn # Sets of preset layers
    import torch.nn.functional as F # Sets of functions such as ReLU
    from torchvision import database, transforms # Popular datasets, architectures and common image transformations for computer vision
```

### Transfrom Dataset
Before we download the data, we will need to specify how we want to transform our dataset. This is a bit different from the Keras's workflow; where we import the dataset then transform the data into the format that we want.

There are two basic transformations that is required for this dataset: turn the raw data into tensor and normalize the data with mean and standard deviation.

As in the example below, we passed 0.5 to both parameters mean and std so that the resulted image could be in the range [-1,1]. See the explanation [here](https://discuss.pytorch.org/t/understanding-transform-normalize/21730/2). <code>Normalization</code> is an important step towards a faster and efficient deep learning model.

Neural network learns how to predict the data by updating its parameters. During training, some features with larger numerical values tend to be assigned with larger parameters. By doing so, we miss the opportunity to learn from other features that could have significant impact on the prediction. Therefore, we need normalization to set every features at the same "starting line" and let the network to decide which feature is important.

```python
    tranform = tranforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
```

### Download Dataset
To download the dataset, we use torchvision dataset library.

```python
    trainset = datasets.MNIST('~/MNIST_data/', download=True, train=True, transform=transform)
```

### Load Dataset
To load the dataset efficiently, we need to utilize the <code>dataloader</code> function.

Normally, when we load data from the dataset, we will naively use <code>forloop</code> to iterate over data. By doing so we are refraining ourselves from:
- Batching the data. Retrieving dataset by batches for mini-batch training
- Shuffling the data. To allow model see different set of training batch in every iteration. This helps to avoid <strong>biased estimate</strong>. See the [explanation](https://www.quora.com/Why-do-we-need-to-shuffle-inputs-for-stochastic-gradient-descent) on why its important to shuffle data.
- Load the data in parallel using <code>multiprocessing</code> workers.

Therefore, we use <code>dataloader</code> to solve the abovementioned issues.












### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://google.com)

Here's a bulleted list:
* First itemprop
+ Second item
- Third item

Here's a numbered list:
1. first
2. second
3. Third

Python code block:
```python
    import numpy as np

    def text_function (x,y):
      return np.sum(x,y)
```
