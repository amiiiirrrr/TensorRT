# TensorRT_keras_model

# Abstract
During the TensorFlow with TensorRT (TF-TRT) optimization, TensorRT performs several important transformations and optimizations to the neural network graph. First, layers with unused output are eliminated to avoid unnecessary computation. Next, where possible, convolution, bias, and ReLU layers are fused to form a single layer. Another transformation is horizontal layer fusion, or layer aggregation, along with the required division of aggregated layers to their respective output. Horizontal layer fusion improves performance by combining layers that take the same source tensor and apply the same operations with similar parameters. This guide provides instructions on how to accelerate inference in TF-TRT.

# Overview
TensorFlow™ integration with TensorRT™ (TF-TRT) optimizes and executes compatible subgraphs, allowing TensorFlow to execute the remaining graph. While you can still use TensorFlow's wide and flexible feature set, TensorRT will parse the model and apply optimizations to the portions of the graph wherever possible.
You will need to create a SavedModel (or frozen graph) out of a trained TensorFlow model (see Build and load a SavedModel), and give that to the Python API of TF-TRT (see Using TF-TRT), which then:
*  returns the TensorRT optimized SavedModel (or frozen graph).
*  replaces each supported subgraph with a TensorRT optimized node (called TRTEngineOp), producing a new TensorFlow graph.

# TensorRT
The core of NVIDIA TensorRT is a C++ library that facilitates high performance inference on NVIDIA graphics processing units (GPUs). TensorRT takes a trained network, which consists of a network definition and a set of trained parameters, and produces a highly optimized runtime engine which performs inference for that network.

# TF-TRT Workflow
The following diagram shows the typical workflow in deploying a trained model for inference.

<img src=https://user-images.githubusercontent.com/28767607/130687566-e2b8ca72-f729-498d-aef6-c735defc3397.PNG width="300" height="200">

In order to optimize the model using TF-TRT, the workflow changes to one of the following diagrams depending on whether the model is saved in SavedModel format or regular checkpoints. Optimizing with TF-TRT is the extra step that is needed to take place before deploying your model for inference.

<img src=https://user-images.githubusercontent.com/28767607/130687640-d039abb8-0b74-491a-a6ec-6c323a8a55b9.PNG width="300" height="200">

# Installing TF-TRT
Compatible Tensorflow, cuda, cudnn and TensorRT versions is needed. To install use below instructions:

```console
foo@bar:~$ whoami

```
