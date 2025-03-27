# Deep Adaptive Fusion Network with Multimodal Neuroimaging Information for MDD Diagnosis: An Open Data Study
This repository contains the code and data for the paper "Deep Adaptive Fusion Network with Multimodal Neuroimaging Information for MDD Diagnosis: An Open Data Study". The code will be made public once the article is accepted！

# Summary
Multimodal neuroimaging provides powerful evidence for automated diagnosis of major depressive disorder (MDD). However, the gap between multimodal neuroimaging information limits the exploration of cross-modal interactions and feature complementarity. In this study, we proposed a supervised deep adaptive fusion network (DAFN) to fully exploit the complementarity of multimodal neuroimaging information for the diagnosis of MDD. Specifically, the high- and low-frequency features in neuroimaging were extracted using a customized convolutional neural network (CNN) and multi-head self-attention encoders (MSA), while the contributions of features from different modalities were explored during training using a modality weight adaptation module (MWAM). Furthermore, we designed a progressive information reinforcement training strategy to enhance adaptive cross-modal information fusion. Finally, the performance of the DAFN was validated on both the open-access dataset and the recruited dataset. The experiment results demonstrate that the proposed DAFN exhibits robust competitiveness in multimodal neuroimaging fusion for MDD diagnosis.

# Dataset
REST-meta-MDD consortium: https://rfmri.org/REST-meta-MDD.
The recruited dataset: https://drive.google.com/drive/folders/1bVTjJMWI6IRAYu rPGiaUDTntwMfJN8Mm?usp=sharing.

# Requirements
The experiments related to this study were compiled using PyTorch-1.13 and executed on an NVIDIA A100 GPU, running on Ubuntu 20.04.
