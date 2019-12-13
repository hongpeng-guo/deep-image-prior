# Deep image prior

In this repository we provide *Jupyter Notebooks* to reproduce each figure from the paper:

> **Deep Image Prior**

> CVPR 2018

> Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky

# Origin
This project is based in the paper Deep Image Prior https://arxiv.org/abs/1711.10925 and code from https://github.com/DmitryUlyanov/deep-image-prior

Authors: Bo Chen, Hongpeng Guo, Xinyang Liu, Yufei Ruan and Zhe Yang.


# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- numpy
- scipy
- matplotlib
- scikit-image
- jupyter

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```


or create an conda env with all dependencies via environment file

```
conda env create -f environment.yml
```

# Citation
```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}
```
