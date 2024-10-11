# Breast Cancer Prediction using TensorFlow (Mini Project)
This mini-project demonstrates how to use TensorFlow to build a simple machine learning model that predicts whether a breast tumor is benign or malignant. The prediction is based on data from the Breast Cancer Wisconsin Dataset, which includes various features extracted from images of breast tumor cells.

# Table of Contents
Overview
Dataset
Installation
Model Architecture
Training the Model
Evaluation
Results
Contributing
License
# Overview
Breast cancer is a significant health issue, and early detection is critical for effective treatment. In this project, we develop a neural network model using TensorFlow to classify breast tumors as benign or malignant based on cell feature measurements. The model aims to support medical professionals in making faster and more accurate diagnoses.

# Dataset
The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, available from the UCI Machine Learning Repository. It contains 30 features derived from digitized images of fine needle aspirates (FNAs) of breast masses.

# Classes: Benign (0) and Malignant (1)
Features: Various attributes such as radius, texture, perimeter, area, smoothness, compactness, etc.

# Model Architecture
The neural network model is built using TensorFlow and consists of:

Input layer: 30 features
Hidden layers: Fully connected layers with ReLU activation functions
Output layer: Single neuron with sigmoid activation for binary classification (benign/malignant)
The loss function used is binary cross-entropy, and the model is optimized using the Adam optimizer.

Training the Model
To train the model, the dataset is split into training and testing sets. The model is trained for several epochs, and the performance is evaluated using accuracy metrics.


# Example of how the model is trained:
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluation
After training, the model's performance is evaluated using the testing set. The key metrics used for evaluation are:

Accuracy
Precision
Recall

These metrics help determine how well the model distinguishes between benign and malignant cases.

# Results
The model achieves a reasonable level of accuracy (approximately 95%) on the testing data. This performance demonstrates the ability of the neural network to assist in breast cancer detection.

# Contributing
This mini-project is for educational purposes. Contributions are welcome for improvements in model architecture, dataset preprocessing, or documentation.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
