# HFO-VAE

This repository contains the implementation of *Discovery of Neurophysiological Characteristics of Pathological High-Frequency Oscillations in Epilepsy with an Explainable Deep Generative Model*

---

## Requirements

All of the code is tested under tested under python 3.9.15 and torch 2.0.0

To install all of the required packages

```
conda create --name hfo --file requirement.txt 
conda activate hfo
```
Then install cuml

```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.6.* dask-cudf-cu11==24.6.* cuml-cu11==24.6.* \
    cugraph-cu11==24.6.* cuspatial-cu11==24.6.* cuproj-cu11==24.6.* \
    cuxfilter-cu11==24.6.* cucim-cu11==24.6.* pylibraft-cu11==24.6.* \
    raft-dask-cu11==24.6.* cuvs-cu11==24.6.*
```

---

## Dataset 

TBD

## Released checkpoint

We have released the model checkpoint for fully reproduction of the reported statistic. Please see ***res/2023-12-28_1*** 

## Training
The model training and architecture hyperparameters could be found in ***src/param.py***
```
python run_training.py
```

## Inference, Charateristic, Metric 

Please run the following commands in order:
Mainly for Figure 2 & Figure 6 and other ablation studies.

```
python run_classification.py
python run_metric.py
python run_characteristic.py
python run_metric_figures.py
```

## Latent space visualization: 

```
python run_plot_embedding.py
```

## Latent space perturbation:

Mainly for Figure 5.

```
python run_perturbation.py
python draw/figure5.py
```



