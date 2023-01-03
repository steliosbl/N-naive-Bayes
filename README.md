# N-naive-Bayes

This is the implementation for the paper titled: *Fairness-aware Naive Bayes Classifier for Data with Multiple Sensitive Features*.

[Publication Link](https://ceur-ws.org/Vol-3276/SSS-22_FinalPaper_69.pdf)

## Table of Contents
------

1. [Introduction](#introduction)
2. [Installation instructions](#installation-instructions)
3. [Repo contents](#repo-contents)
4. [References](#references)

## Introduction
N-naive-Bayes (or NNB) is a fairness-aware machine learning model. It is a generalisation of Two-naive-Bayes, a simple yet effective group-fair binary classification algorithm that which enforces statistical parity - the requirement that the groups comprising the dataset receive positive labels with the same likelihood [1]. 

In the paper for this project, we generalise this algorithm to eliminate the simplification of assuming only two sensitive groups in the data and instead apply it to an arbitrary number of groups. We propose an extension of the original algorithm's statistical parity constraint and the post-processing routine that enforces statistical independence of the label and the single sensitive attribute. Then, we investigate its application on data with multiple sensitive features and propose a new constraint and post-processing routine to enforce *differential fairness*, an extension of established group-fairness constraints focused on intersectionalities [2]. 

This repository contains the scikit-Learn implementation of the proposed algorithm. The Jupyter notebook at the root directory contains the tests we ran to demonstrate the algorithm's debiasing performance and accuracy on US Census datasets.

## Installation instructions
------
1. (Optionally) create a virtual environment
```
python3 -m venv folkenv
source folkenv/bin/activate
```
2. Clone into directory
```
git clone https://github.com/stelioslogothetis/N-naive-Bayes.git
cd N-naive-Bayes
```
3. Install requirements via pip
```
pip install -r requirements.txt
```

### Requirements:

 - Sklearn >= 0.24.2
 - [Folktables](https://github.com/zykls/folktables) (available on pip)
 - NumPy
 - Pandas
 - Matplotlib
 - Seaborn
 - Jupyter

## Repo contents
------

The notebook `n_naive_bayes.ipynb` demonstrates the testing for the N-naive-Bayes project. 

The source for the classifiers can be found in the `models/` directory.
 - `models/nnb_base`: The base class for NNB, containing fitting and prediction generation
 - `models/nnb_parity`: The statistical parity version of NNB
 - `models/nnb_df`: The differential fairness version of NNB
 - `models/two_naive_bayes`: A scikit-Learn implementation of the original CV2NB
 - `models/gaussian_sub`: The Gaussian naive Bayes sub-estimator

Supporting code:
 - `dataset.py`: Classes for interacting with the US Census data used for testing. See the [folktables library](https://github.com/zykls/folktables)
 - `scoring.py`: Scoring functions implementing various popular group-fairness measures

## References
------
[1] Toon Calders and Sicco Verwer. 2010. Three naive Bayes approaches for discrimination-free classification. Data Mining and Knowledge Discovery 21,
2 (01 Sep 2010), 277–292. https://doi.org/10.1007/s10618-010-0190-x

[2] James R. Foulds, Rashidul Islam, Kamrun Naher Keya, and Shimei Pan. 2020. An Intersectional Definition of Fairness. In 2020 IEEE 36th International
Conference on Data Engineering (ICDE). IEEE, Dallas, Texas, USA. https://doi.org/10.1109/ICDE48307.2020.00203
