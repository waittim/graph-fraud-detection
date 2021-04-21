# Fraud Detection in Graph Neural Network

Many online businesses lose billions of dollars to fraud each year, but machine learning-based fraud detection models can help businesses predict which interactions or users are likely to be fraudulent in order to reduce losses.

This repo formulates the problem of fraud detection as a classification task for heterogeneous interaction networks. The machine learning model used is a graphical neural network (GNN) that learns potential representations of users or transactions, which can then be easily classified as Fraud or not.

The model used in this repo is refactored from the model used in [awslabs/sagemaker-graph-fraud-detection](https://github.com/awslabs/sagemaker-graph-fraud-detection), and implemented based on [DGL](https://github.com/dmlc/dgl) and PyTorch.

Unlike Amazon's implementation, this repo does not require the use of Sagemaker for training. We can do it directly with the free Google Colab or with our own local devices.

## Usage

### 1. Download dataset

First, we need to download [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) data from Kaggle. This [link](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203) provided some additional information about the dataset.

Then put all of the CSV files into the `./ieee-data` fold.

### 2. Data preparation

Before feeding the data to the model, we need to perform data pre-processing. Open **10_data_loader.ipynb** and follow the introduction inside. The compiled data will be saved into the `./data` fold.

### 3. Training

Open **20_modeling_pytoch.ipynb** and follow the introduction inside. 

### 4. After training

The trained models and related files will be save into the `./model` fold.
