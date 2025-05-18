# Deep Adaptive Fusion Network with Multimodal Neuroimaging Information for MDD Diagnosis: An Open Data Study
This repository contains the code and data for the paper "Deep Adaptive Fusion Network with Multimodal Neuroimaging Information for MDD Diagnosis: An Open Data Study". **The code will be made public once the article is accepted！**

![maige](https://github.com/TTLi1996/DAFN/blob/main/Overview%20of%20the%20DAFN.jpg)

# Summary
Neuroimaging provides powerful evidence for the automated diagnosis of major depressive disorder (MDD). Nevertheless, disparities between different imaging modalities limit the exploration of cross-modal interactions and hinder the effective integration of complementary features. To bridge this gap, we propose a supervised Deep Adaptive Fusion Network (DAFN) to fully leverage the complementarity of multimodal neuroimaging information for the diagnosis of MDD. Specifically, high- and low-frequency features are extracted from the images using a customized convolutional neural network and multi-head self-attention encoders, respectively, while a modality weight adaptation module dynamically optimizes the contribution of each modality during training. Furthermore, we design a progressive information reinforcement training strategy to reinforce the multimodal fusion features. Finally, the performance of the DAFN is evaluated on both the open-access dataset and the recruited dataset. The results show that DAFN achieves competitive performance in multimodal neuroimaging fusion for the diagnosis of MDD.

# Dataset
REST-meta-MDD consortium: https://rfmri.org/REST-meta-MDD.

The recruited dataset: https://drive.google.com/drive/folders/1bVTjJMWI6IRAYurPGiaUDTntwMfJN8Mm?usp=sharing.

# Requirements
The experiments related to this study were compiled using TensorFlow-2.6.0 and Keras-2.6.0, and executed on an NVIDIA A100 GPU, running on Ubuntu 20.04.
