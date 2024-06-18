# Credit Card Fraud Detection

## Introduction

This Python-based project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The goal is to identify fraudulent transactions accurately to prevent financial loss and ensure the security of credit card users. The project utilizes various classification techniques to achieve this objective.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
   - [Random Forest Classifier](#random-forest-classifier)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
   - [Gradient Boosted Decision Tree (GBT)](#gradient-boosted-decision-tree-gbt)
4. [Conclusion](#conclusion)

## Overview

This project utilizes a dataset containing credit card transactions, focusing on detecting fraudulent transactions among legitimate ones. It employs various machine learning classifiers such as Random Forest, SVM, KNN, and GBT to classify transactions accurately.

## Installation

**IMPORTANT: Downloading Data**:
Download File (.zip) from the drive and unzip the file in the same folder. To use in the program.
Download the dataset from drive: https://drive.google.com/file/d/1lfNNhR14reZK_N05J0abN9TJEI8w4ayc/view?usp=sharing
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. **Navigate to the Project Directory**:
   ```sh
   cd credit-card-fraud-detection
   ```
3. **Install Dependencies**:
   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

## Usage

### Data Exploration and Preprocessing

```python
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Explore the dataset
print(data.head())
print(data.describe())
print(data.shape)

# Check outlier fraction
frauds = data[data["Class"] == 1]
valids = data[data["Class"] == 0]
outlierFRAC = len(frauds) / float(len(valids))
print("Outlier Fraction : ", outlierFRAC)

# Normalize data if necessary
if outlierFRAC > 0.1:
    data = pd.DataFrame(preprocessing.normalize(data))

# Correlation matrix
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Split data into training and testing sets
X = data.drop(['Class'], axis=1)
Y = data["Class"]
xData = X.values
yData = Y.values
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=39)
```

### Random Forest Classifier

```python
RFC = RandomForestClassifier()
RFC.fit(xTrain, yTrain)
ypredicted = RFC.predict(xTest)

acc_rf = accuracy_score(yTest, ypredicted)
rf_prec = precision_score(yTest, ypredicted)
rf_rec = recall_score(yTest, ypredicted)
rf_f1 = f1_score(yTest, ypredicted)

# Print metrics and confusion matrix
print("Random Forest Classifier Metrics:")
print("Accuracy:", acc_rf)
print("Precision:", rf_prec)
print("Recall:", rf_rec)
print("F1 Score:", rf_f1)
print("Confusion Matrix:")
print(confusion_matrix(yTest, ypredicted))
```

### Support Vector Machine (SVM)

```python
clf = LinearSVC()
clf.fit(xTrain, yTrain)
ypredicted = clf.predict(xTest)

acc_svm = accuracy_score(yTest, ypredicted)
svm_prec = precision_score(yTest, ypredicted)
svm_rec = recall_score(yTest, ypredicted)
svm_f1 = f1_score(yTest, ypredicted)

# Print metrics and confusion matrix
print("Support Vector Machine (SVM) Metrics:")
print("Accuracy:", acc_svm)
print("Precision:", svm_prec)
print("Recall:", svm_rec)
print("F1 Score:", svm_f1)
print("Confusion Matrix:")
print(confusion_matrix(yTest, ypredicted))
```

### K-Nearest Neighbors (KNN)

```python
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(xTrain, yTrain)
ypredicted = knn.predict(xTest)

acc_knn = accuracy_score(yTest, ypredicted)
knn_prec = precision_score(yTest, ypredicted)
knn_rec = recall_score(yTest, ypredicted)
knn_f1 = f1_score(yTest, ypredicted)

# Print metrics and confusion matrix
print("K-Nearest Neighbors (KNN) Metrics:")
print("Accuracy:", acc_knn)
print("Precision:", knn_prec)
print("Recall:", knn_rec)
print("F1 Score:", knn_f1)
print("Confusion Matrix:")
print(confusion_matrix(yTest, ypredicted))
```

### Gradient Boosted Decision Tree (GBT)

```python
gbct = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=40)
gbct.fit(xTrain, yTrain)
ypredicted = gbct.predict(xTest)

acc_gbt = accuracy_score(yTest, ypredicted)
gbt_prec = precision_score(yTest, ypredicted)
gbt_rec = recall_score(yTest, ypredicted)
gbt_f1 = f1_score(yTest, ypredicted)

# Print metrics and confusion matrix
print("Gradient Boosted Decision Tree (GBT) Metrics:")
print("Accuracy:", acc_gbt)
print("Precision:", gbt_prec)
print("Recall:", gbt_rec)
print("F1 Score:", gbt_f1)
print("Confusion Matrix:")
print(confusion_matrix(yTest, ypredicted))
```

## Conclusion

This project showcases the application of various machine learning classifiers for credit card fraud detection. Each classifier is evaluated based on accuracy, precision, recall, and F1-score. The choice of classifier depends on specific requirements such as performance and interpretability. Further optimizations and fine-tuning of models can enhance detection accuracy and efficiency in real-world scenarios.

## Contributing

Contributions are welcome. Fork the repository and submit a pull request with your changes. Ensure your code follows the project's standards and includes appropriate tests.

---

For further assistance, please contact <tanm280604@gmail.com>.

Be Aware, Stay Safe!
