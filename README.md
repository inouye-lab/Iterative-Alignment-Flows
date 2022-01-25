# Iterative Alignment Flows



## Installation

Our models are based on scikit-learn and PyTorch.

## Modules

It is not required to install other modules to run our codes.

Folder weakflow, ddl, SINF and python files in the main folder include necessary functions to implement our experiments.

## Implementation

Folder demos include several notebooks that implement our experiments. To run the notebooks, you need to modify the sys.path. You also need to comment out the codes where we load the data and uncomment the codes where we make the data. (Since the truncated data can vary, the generated samples can vary from the paper.)

For each experiment of SINF, there is a separate demo.

demo_convergence_2D.ipynb is a separate notebook for Figure1.(e) in the main paper.

As for the implementation of AlignFlow experiment, please read the README.md file in the folder alignflow.

(Some outputs are cleared for anonymity and space limit)

