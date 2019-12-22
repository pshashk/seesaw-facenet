## PyTorch implementation of [Seesaw-shuffleFaceNet](https://arxiv.org/abs/1908.09124)

Differences from [official implementation](https://github.com/cvtower/seesawfacenet_pytorch/blob/master/src/seesaw_models/seesaw_shuffleFaceNet.py) are minimal:
 - [Li-ArcFace loss](https://arxiv.org/abs/1907.12256)
 - [Zero γ](https://arxiv.org/abs/1812.01187)
 - standard SE Block (no bias, no prelu)
 - ReLU after first two convolutions
 - Hard versions for Swish and Sigmoid

[Pretrained model (TorchScript)](https://drive.google.com/file/d/1Ub5CI3nqTekLnG1AH1cGrQcwblW5YWoa/)

Validation metrics can be computed by running `python3 eval.py <path to model> <path to a folder containing InsightFace bin files>`:
```
cfp_ff: 99.56%
lfw: 99.62%
agedb_30: 96.12%
vgg2_fp: 93.24%
cfp_fp: 94.49%
cplfw: 89.52%
```

For training code see [foamliu/InsightFace-v3](https://github.com/foamliu/InsightFace-v3) and similar projects.

Tested with nightly builds of PyTorch (`1.4.0.dev20191216`).