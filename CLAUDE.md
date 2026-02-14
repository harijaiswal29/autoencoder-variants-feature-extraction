# Assignment 01 - Advanced Deep Learning

## Project Overview

This project implements **feature extraction via dimensionality reduction using autoencoder variants** on the CIFAR-10 and MNIST datasets. It is Assignment 01 for the Advanced Deep Learning (ADL) course at BITS Pilani (M.Tech AI/ML program).

## Team

- **Group 04** (ADL)
- Warun Kumar (2023aa05244)
- Gandhi Disha Dipak Shaila (2023aa05388)
- Hari Shankar Jaiswal (2023aa05106)
- Jaideep Dave (2023ab05021)

## Structure

```
Assignment 01/
  MarkdownFile.md                  # Problem statement
  Assignment Problem Statement.docx
  Final/
    ADL_Group_04_Assignment01.ipynb # Main solution notebook
    ADL_Group_04_Assignment01.html  # Exported HTML with outputs
    Uploaded Version/               # Version submitted for grading
```

## Tasks

### Task 1: PCA + Logistic Regression
- Dataset: CIFAR-10 grayscale, 70/30 train-test split
- Standard PCA with 95% variance retention -> Logistic Regression classifier (10 classes)
- ROC curves and AUC computation
- Comparison with Randomized PCA
- Key result: Both standard and randomized PCA yield ~0.73 micro-average AUC

### Task 2: Single-Layer Tied-Weight Linear Autoencoder
- Linear activation, encoder/decoder weights are transposes of each other
- Mean-variance normalized input
- Comparison of learned weights with PCA eigenvectors
- Key result: ~0.989 average cosine similarity with PCA components

### Task 3: Deep Convolutional Autoencoder
- CNN autoencoder on CIFAR-10 grayscale (latent dim = 64)
- Reconstruction error comparison: CNN AE vs single-layer AE
- Key result: CNN MSE ~4.3e-6 vs single-layer MSE ~0.054

### Task 4: MNIST 7-Segment Classification
- CNN autoencoder on MNIST for feature extraction (latent dim = 128)
- MLP classifier mapping latent features to 7-segment LED outputs
- Confusion matrix evaluation

## Tech Stack

- Python, NumPy, Matplotlib
- TensorFlow/Keras (Conv2D, Conv2DTranspose, Dense layers)
- scikit-learn (PCA, LogisticRegression, metrics)

## How to Run

Open `Final/ADL_Group_04_Assignment01.ipynb` in Jupyter/Colab and run all cells sequentially. The notebook downloads CIFAR-10 and MNIST automatically via `tensorflow.keras.datasets`.
