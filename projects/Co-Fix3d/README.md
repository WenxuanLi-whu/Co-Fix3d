# Co-Fix3D: Enhancing 3D Object Detection With Collaborative Refinement

https://arxiv.org/pdf/2408.07999

<!-- [ALGORITHM] -->

## Abstract

3D object detection in driving scenarios faces the challenge of complex road environments, which can lead to the loss or incompleteness of key features, thereby affecting perception performance. To address this issue, we propose an advanced detection framework called Co-Fix3D. Co-Fix3D integrates Local and Global Enhancement (LGE) modules to refine Bird's Eye View (BEV) features. The LGE module uses Discrete Wavelet Transform (DWT) for pixel-level local optimization and incorporates an attention mechanism for global optimization. To handle varying detection difficulties, we adopt multi-head LGE modules, enabling each module to focus on targets with different levels of detection complexity, thus further enhancing overall perception capability. Experimental results show that on the nuScenes dataset's LiDAR benchmark, Co-Fix3D achieves 69.4\% mAP and 73.5\% NDS, while on the multimodal benchmark, it achieves 72.3\% mAP and 74.7\% NDS.
<div align=center>
<img src="https://github.com/user-attachments/assets/1badfdaa-0c7e-4584-a1fb-cc36b3a913b4" width="800"/>
</div>

## Introduction

We implement Co-Fix3D and support training and testing on NuScenes dataset.

