# ADL Assignment 01 - Feature Extraction via Dimensionality Reduction Using Autoencoders

**Course:** Advanced Deep Learning (ADL) | BITS Pilani M.Tech AI/ML | Semester 3
**Group:** 04

| Name | BITS ID |
|------|---------|
| Warun Kumar | 2023aa05244 |
| Gandhi Disha Dipak Shaila | 2023aa05388 |
| Hari Shankar Jaiswal | 2023aa05106 |
| Jaideep Dave | 2023ab05021 |

## Problem Statement

This assignment explores feature extraction through dimensionality reduction using various autoencoder architectures. The work is divided into four tasks, using CIFAR-10 (converted to grayscale) for Tasks 1-3 and MNIST for Task 4. All datasets use a 70% train / 30% test split.

## Tasks and Results

### Task 1: PCA-Based Feature Extraction and Classification

**Objective:** Apply Standard PCA and Randomized PCA on CIFAR-10 grayscale images, retain components explaining 95% of total variance, and train a Logistic Regression classifier on the reduced features.

**Approach:**
- Standard PCA (`svd_solver='full'`) with `n_components=0.95` to automatically select the number of principal components covering 95% variance
- One-vs-Rest Logistic Regression trained on PCA-reduced features
- ROC curves plotted per class and micro-averaged

**Results:**
| Method | Micro-Average AUC |
|--------|-------------------|
| Standard PCA | 0.73 |
| Randomized PCA | 0.73 |

**Observation:** Both PCA variants produce nearly identical classification performance, confirming that randomized PCA captures the principal variance directions as effectively as exact PCA for this dataset. The moderate AUC reflects the inherent difficulty of classifying CIFAR-10 with a linear model on raw pixel features.

### Task 2: Single-Layer Linear Autoencoder vs PCA

**Objective:** Train a single-layer autoencoder with linear activation and tied encoder-decoder weights (W_decoder = W_encoder^T), then compare the learned weight vectors with PCA eigenvectors.

**Approach:**
- Custom `TiedAutoencoder` Keras layer enforcing W_decoder = W_encoder^T during the forward pass
- Input is mean-variance normalized
- Latent dimension matches the number of PCA components from Task 1
- Trained for 50 epochs with Adam (lr=1e-4), MSE loss
- Optionally initialized with PCA components for faster convergence

**Results:**
| Metric | Value |
|--------|-------|
| Training MSE | ~0.051 |
| Validation MSE | ~0.052 |
| Avg. Cosine Similarity with PCA Eigenvectors | 0.989 |

**Observation:** The ~0.989 cosine similarity confirms the theoretical equivalence: a linear autoencoder with tied weights learns a subspace that closely aligns with the PCA principal subspace. Key enabling factors include strict weight tying, proper mean-centering, and sufficient training.

### Task 3: Deep Convolutional Autoencoder vs Shallow Autoencoders

**Objective:** Build a deep convolutional autoencoder (CNN AE) for CIFAR-10 grayscale and compare its reconstruction error with the single-layer autoencoder from Task 2.

**Architecture (CNN Autoencoder):**
- **Encoder:** Conv2D(32) -> MaxPool -> Conv2D(16) -> MaxPool -> Flatten -> Dense(64)
- **Decoder:** Dense(1024) -> Reshape(8,8,16) -> Conv2DTranspose(16) -> Conv2DTranspose(32) -> Conv2D(1, sigmoid)
- Latent dimension: 64

**Results:**
| Model | Test MSE |
|-------|----------|
| CNN Autoencoder | 4.34e-06 |
| Single-Layer Linear AE | 0.054 |

**Observation:** The CNN autoencoder achieves reconstruction error orders of magnitude lower than the single-layer AE, demonstrating the power of convolutional feature hierarchies for image data. The sigmoid output layer paired with [0,1] normalized inputs enables near-perfect pixel-level reconstruction.

### Task 4: MNIST Feature Extraction + 7-Segment LED Classification

**Objective:** Train a deep convolutional autoencoder on MNIST, extract latent features, and use an MLP to classify digits into their 7-segment LED display representations (7 binary outputs).

**7-Segment Encoding:**
```
Segment mapping: [a, b, c, d, e, f, g]
0 -> [1,1,1,1,1,1,0]    5 -> [1,0,1,1,0,1,1]
1 -> [0,1,1,0,0,0,0]    6 -> [1,0,1,1,1,1,1]
2 -> [1,1,0,1,1,0,1]    7 -> [1,1,1,0,0,0,0]
3 -> [1,1,1,1,0,0,1]    8 -> [1,1,1,1,1,1,1]
4 -> [0,1,1,0,0,1,1]    9 -> [1,1,1,1,0,1,1]
```

**Architecture:**
- **Autoencoder:** Conv2D(32) -> MaxPool -> Conv2D(16) -> MaxPool -> Flatten -> Dense(128) [latent] -> Dense(784) -> Reshape -> Conv2DTranspose layers -> Conv2D(1, sigmoid)
- **MLP Classifier:** Dense(128, relu) -> Dense(64, relu) -> Dense(32, relu) -> Dense(7, sigmoid)

**Approach:**
- Autoencoder trained for 30 epochs on MNIST (28x28x1)
- Encoder extracts 128-dimensional latent features
- MLP trained with binary cross-entropy loss on 7-segment targets
- Predictions thresholded at 0.5 and mapped back to digit labels
- Confusion matrix generated on test set

**Observation:** The autoencoder's latent features are discriminative enough for the MLP to learn the digit-to-7-segment mapping with high accuracy. Misclassifications tend to occur between digits with similar segment patterns (e.g., 8 vs 9 differ by only one segment).

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow / Keras
- scikit-learn

## How to Run

1. Open `Final/ADL_Group_04_Assignment01.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially (datasets are downloaded automatically)
3. The HTML export with all embedded outputs is available at `Final/ADL_Group_04_Assignment01.html`

## File Structure

```
Assignment 01/
├── README.md                              # This file
├── CLAUDE.md                              # Project context for Claude Code
├── MarkdownFile.md                        # Problem statement (markdown)
├── Assignment Problem Statement.docx      # Problem statement (original)
└── Final/
    ├── ADL_Group_04_Assignment01.ipynb     # Solution notebook
    ├── ADL_Group_04_Assignment01.html      # HTML export with outputs
    └── Uploaded Version/                   # Submitted version
        ├── ADL_Group_04_Assignment01.ipynb
        └── ADL_Group_04_Assignment01.html
```
