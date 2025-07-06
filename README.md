# 🎨 ANN-Based Color Detection System

This project implements a Color Detection System using an Artificial Neural Network (ANN) to predict color names from RGB values. It classifies 173 unique color names based on a Kaggle dataset, using TensorFlow, Keras, and Scikit-learn for model development and evaluation.

# Features

- Predict color names from RGB input
- Built with TensorFlow and Keras
- Trained on 173 color classes from Kaggle dataset
- One-hot encoded labels for multi-class classification
- Visual performance evaluation (accuracy/loss, confusion matrix)
- Top-3 prediction display with confidence scores
- User interface for image upload and pixel click detection

# Model Architecture

- Input Layer: 3 Neurons (RGB)
- Hidden Layers: 64 and 128 Neurons (ReLU)
- Output Layer: 173 Neurons (Softmax)
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

# Tools & Libraries

- Python 3.10.9
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit (for UI deployment)

# Dataset

- **Source**: [Kaggle Color Names Dataset](https://www.kaggle.com/datasets/paulvangentcom/color-names)
- **Preprocessing**:
  - Cleaned and renamed RGB columns
  - Removed unnecessary fields
  - One-hot encoded color names

# Evaluation Results

- Accuracy Score: 0.000 (Model overfit due to class imbalance)
- Confusion Matrix: Sparse, highlights class confusion
- F1-Score & Precision: Near 0 due to overfitting
- Training Accuracy: 35%+
- Validation Accuracy: 0%
- Observed overfitting in loss/accuracy graphs

# Screenshots

- Model architecture code
- Accuracy & loss plot
- Confusion matrix heatmap
- Classification report
- Streamlit UI preview


# How to Run

# Train the model
python ANN_model.py

# Run the Streamlit UI
streamlit run app.py

