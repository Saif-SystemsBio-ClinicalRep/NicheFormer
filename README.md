# NicheFormer
NicheFormer: A Spatial Graph Transformer and Causal Inference Framework Decodes COL1A1-Driven Desmoplasia and Evolutionary Dynamics in Human Breast Carcinoma
# NicheFormer: Spatial Graph Transformers & Causal AI Framework

This repository contains the core computational architecture and mathematical framework supporting the manuscript: **"NicheFormer: A Spatial Graph Transformer and Causal Inference Framework Decodes COL1A1-Driven Desmoplasia and Evolutionary Dynamics in Human Breast Carcinoma."**

## Overview
NicheFormer is a disease-agnostic computational framework designed to transition spatial transcriptomics from descriptive clustering to predictive biophysics. It provides the architectural classes to fuse Graph Neural Networks (GNNs) with Causal Inference (Graphical Lasso) and thermodynamic modeling.

## Repository Structure
Due to the massive file sizes of high-resolution 10x Genomics Visium spatial tensors (>10GB) and the heavy GPU compute requirements for training the spatial transformers, this repository hosts the **core methodological framework and architectural classes** used to execute the pipeline.

* `nicheformer_architecture.py`: Contains the PyTorch Geometric class definition for the multi-head spatial self-attention network and the XAI extraction methods.
* `spatial_physics_engine.py`: Contains the statistical and mathematical engines for Causal GRN construction, Shannon Entropy, and Kaplan-Meier digital translation.
* `requirements.txt`: Environment dependencies.


```python
from nicheformer_architecture import NicheFormer
from spatial_physics_engine import SpatialBiophysics

# Initialize the architecture for your dataset
model = NicheFormer(in_channels=YOUR_GENE_COUNT, hidden_channels=64, out_channels=1)
