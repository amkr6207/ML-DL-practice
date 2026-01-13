# Digit Recognizer - Kaggle Competition

This project implements a deep learning solution for the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer). The goal is to classify handwritten digits (0-9) from the famous MNIST dataset.

## Competition Overview
- **Question**: [Digit Recognizer Overview](https://www.kaggle.com/competitions/digit-recognizer/overview)
- **Dataset**: 42,000 training images and 28,000 test images (28x28 grayscale pixels).
- **Goal**: Predict the correct label for each test image.

## Solution Approach

### 1. Data Exploration (EDA)
- Loaded `train.csv` and `test.csv`.
- Visualized image data to verify label consistency.

### 2. Methodology & Modeling
- **Preprocessing**: Normalized pixel values to [0, 1] and split into training and validation sets.
- **Architecture**: Implemented a Convolutional Neural Network (CNN) using PyTorch:
  - **Conv Layers**: Two convolution layers with ReLU activation and Max Pooling to extract spatial features.
  - **Dense Layers**: Fully connected layers with Dropout to prevent overfitting.
- **Optimization**: Adam optimizer with Cross-Entropy loss.

### 3. Training & Validation
- Trained for 5 epochs.
- Achieved a **Local Validation Accuracy of ~98.5%**.

## Files in this Repository
- `digit_recognizer.ipynb`: Implementation notebook with code and comments.
- `submission.csv`: Final predictions formatted for Kaggle upload.
- `data/`: Competition datasets.

## How to Run
1. Activate your environment (e.g., `conda activate hello-env`).
2. Run the `digit_recognizer.ipynb` notebook using Jupyter.
