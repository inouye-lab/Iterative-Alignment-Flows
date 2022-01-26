# Iterative Alignment Flows

## Reference

This is the official implementation of [Iterative Alignment Flows](https://arxiv.org/abs/2104.07232)

## Installation

Our models are based on scikit-learn and PyTorch.

## Requirements

ddl can be found in the submodule destructive-deep-learning

## Implementation

Folder demos include several notebooks that implement our experiments. To run the notebooks, you need to modify the sys.path. You also need to comment out the codes where we load the data and uncomment the codes where we make the data. (Since the truncated data can vary, the generated samples can vary from the paper.)

demo_convergence_2D.ipynb is a separate notebook for Figure1.(e) in the main paper.
