# Point Spatio-Temporal Transformer Networks for Point Cloud Video Modeling

## Introduction
Due to the inherent unorderliness and irregularity of point cloud, points emerge inconsistently across different frames in a point cloud video. To capture the dynamics in point cloud videos, tracking points and limiting temporal modeling range are usually employed to preserve spatio-temporal structure. However, as points may flow in and out across frames, computing accurate point trajectories is extremely difficult, especially for long videos. Moreover, when points move fast, even in a small temporal window, points may still escape from a region. Besides, using the same temporal range for different motions may not accurately capture the temporal structure. In this paper, we propose a Point Spatio-Temporal Transformer (PST-Transformer). To preserve the spatio-temporal structure, PST-Transformer adaptively searches related or similar points across the  entire video by performing self-attention on point features. Moreover, our PST-Transformer is equipped with an ability to encode spatio-temporal structure. Because point coordinates are irregular and unordered but point timestamps exhibit regularities and order, the spatio-temporal encoding is decoupled to reduce the impact of the spatial irregularity on the temporal modeling. By properly preserving and encoding spatio-temporal structure, our PST-Transformer effectively models point cloud videos and shows superior performance on 3D action recognition and 4D semantic segmentation.

## Installation
The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 9.4.0, PyTorch (v1.10.2), CUDA 11.3.1 and cuDNN v8.2.0.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
cd modules
python setup.py install
```
