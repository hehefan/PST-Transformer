# PST-Transformer

The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 8.3.1, PyTorch (both v1.4.0 and v1.9.0 are supported), CUDA 10.2 and cuDNN v7.6.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
mv modules-pytorch-1.4.0/modules-pytorch-1.9.0 modules
cd modules
python setup.py install
```
