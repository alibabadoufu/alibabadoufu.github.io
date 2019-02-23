---
layout: single
title: "MNIST Classifier with Pytorch [Part I]"
date: 2018-01-28
tags: [machine learning, pytorch, neural network]
excerpt: "Machine Learning, Programming, Pytorch"
comments: true
toc: true
---

I am doing a revision on how to build neural network with PyTorch. To do so I am taking Udacity's online lesson on [Intro to Deep Learning with PyTorch] (https://www.udacity.com/course/deep-learning-pytorch--ud188). This course introduces many important models such as CNN and RNN using PyTorch. In the subsequent posts, I will try to summarize important points that I have learnt in the course.

## Prerequisite
For those who have learnt fundamental deep learning concepts should not find any difficulties to follow. However, if you are a fresh new beginner in this field, I would strongly encourage you to go through Andrew Ng's Deep Learning Specialization on Coursera before reading this entire post series.

## What is MNIST Dataset?
MNIST consists of greyscale handwritten digits ranging from 0 to 9. Each image is 28 x 28 pixels.

<figure>
<a href="https://corochann.com/wp-content/uploads/2017/02/mnist_plot.png"><img src="https://corochann.com/wp-content/uploads/2017/02/mnist_plot.png"></a>
<figcaption>MNIST</figcaption>
</figure>

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
