# Iterative Alignment Flows



## Installation

Our models are based on scikit-learn and PyTorch.

## Requirements

ddl can be found in the submodule destructive-deep-learning

## Implementation

Folder demos include several notebooks that implement our experiments. To run the notebooks, you need to modify the sys.path. You also need to comment out the codes where we load the data and uncomment the codes where we make the data. (Since the truncated data can vary, the generated samples can vary from the paper.)

demo_convergence_2D.ipynb is a separate notebook for Figure1.(e) in the main paper.

## Reference

If you find this code helpful, we would be grateful if you cite the following 

```markdown
@inproceedings{zhou2022align,
  title = { Iterative Alignment Flows },
  author = {Zhou, Zeyu and Gong, Ziyu and Ravikumar, Pradeep and Inouye, David I.},
  booktitle = {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  year = {2022}
}
```

## Update

01/25/2022: At this point, mSWD-NB in the demos refer to INB.