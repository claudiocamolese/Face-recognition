# Face Recognition with SVD and K-Nearest Classifier

This repository implements a face recognition system using **Singular Value Decomposition (SVD)** for dimensionality reduction and a **K-Nearest Classifier (KNC)** for identity recognition. The approach is inspired by the classical **eigenfaces** technique and provides an effective way to handle high-dimensional facial image data.


## 🔍 Overview

Face recognition is a task in computer vision where the goal is to identify or verify a person from a facial image. In this project:

1. Each facial image is flattened into a 1D vector.
2. The dataset is decomposed using **SVD**, projecting images into a low-dimensional subspace that captures the most relevant features.
3. A **K-Nearest Classifier** is trained on the reduced representations to classify new unseen faces.

This combination provides both efficiency (thanks to dimensionality reduction) and flexibility (thanks to the non-parametric nature of KNC).


## 🧠 Singular Value Decomposition (SVD)

Singular Value Decomposition is a matrix factorization technique used to reduce data dimensionality. It factorizes the data matrix as:

$$
A = U \Sigma V^T
$$

Where:
- $A$ is the data matrix (each row is a flattened face image),
- $U$ contains the **left singular vectors** (capturing the principal directions of variation across images),
- $\Sigma$ is a diagonal matrix with **singular values** (ordered by importance),
- $V^T$ contains the **right singular vectors** (basis of the feature space).

### Why SVD is Important in Face Recognition

- **Dimensionality Reduction**: Facial images often have thousands of pixels. SVD reduces the dimensionality while preserving most of the variance.
- **Noise Filtering**: By retaining only the top $k$ singular values, the algorithm filters out noise and irrelevant details.
- **Foundation for Eigenfaces**: The leading components of $U$ are known as eigenfaces — they form a basis for face space representation.
- **Efficiency**: It significantly reduces computation time for training and classification.


## 📊 Classification with K-Nearest Classifier (KNC)

After SVD, faces are represented as low-dimensional vectors in feature space. Classification is performed using the **K-Nearest Classifier**, which assigns a label to a new image based on the majority label among its $k$ closest training samples.

### Key Points:
- **Non-parametric**: No model is explicitly learned; the algorithm memorizes the training data.
- **Simple and effective**: Works well with compact representations, such as those obtained through SVD.
- **Hyperparameter**: The value of $k$ can be tuned to control bias-variance tradeoff.
