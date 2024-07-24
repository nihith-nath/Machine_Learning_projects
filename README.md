
# Machine Learning Projects

Welcome to my Machine Learning GitHub repository. This repository contains detailed reports and projects that I have worked on, showcasing various machine learning techniques and applications. Below is a summary of the key projects included in this repository.

## Table of Contents

1. [Clustering Analysis](#clustering-analysis)
2. [MNIST Dataset Classification](#mnist-dataset-classification)
3. [California Housing Price Prediction](#california-housing-price-prediction)

---

## Clustering Analysis

### Introduction
This project focuses on clustering analysis using various clustering algorithms to group similar data points. The objective is to explore and understand the underlying structure of the data by segmenting it into meaningful clusters.

### Methodology
- **Algorithms Used**: K-means, Hierarchical Clustering, DBSCAN
- **Data Preprocessing**: Standardization, Dimensionality Reduction (PCA)
- **Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index

### Key Findings
- K-means clustering identified clear segments within the data with a Silhouette Score of `X`.
- Hierarchical clustering provided a dendrogram that helped in understanding the data hierarchy.
- DBSCAN was effective in identifying outliers and noise within the dataset.

### Conclusion
Clustering analysis revealed significant patterns and groupings in the data, providing insights for further analysis and application.

---

## MNIST Dataset Classification

### Introduction
The MNIST dataset classification project aims to build and evaluate models to classify handwritten digits from 0 to 9. This project explores different machine learning algorithms and their performance on image data.

### Methodology
- **Dataset**: MNIST handwritten digit dataset
- **Algorithms Used**: Logistic Regression, SVM, Random Forest, Convolutional Neural Networks (CNN)
- **Data Preprocessing**: Normalization, Image Augmentation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

### Key Findings
- **Logistic Regression**: Achieved an accuracy of `X%` with basic preprocessing.
- **SVM**: Improved accuracy to `Y%` with kernel tricks.
- **Random Forest**: Provided robust performance with an accuracy of `Z%`.
- **CNN**: Achieved the highest accuracy of `W%`, leveraging deep learning techniques.

### Conclusion
The CNN model significantly outperformed traditional machine learning models, demonstrating the power of deep learning in image classification tasks.

---

## California Housing Price Prediction

### Introduction
This project involves predicting median house values in California based on the 1990 census data. The objective is to explore various regression models and determine the best approach for accurate price prediction.

### Methodology
- **Dataset**: California Housing dataset from the 1990 census
- **Data Features**: Median income, house age, average number of rooms, latitude, longitude, etc.
- **Models Used**: Simple Linear Regression, Multiple Linear Regression, Locally Weighted Linear Regression (LWLR)
- **Evaluation Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score

### Key Findings
1. **Exploratory Data Analysis (EDA)**:
   - No missing values were found in the dataset.
   - Median income showed a strong positive correlation with median house value.
   - Geographical patterns indicated higher house values in coastal areas.

2. **Simple Linear Regression**:
   - MAE: `0.617` (Train), `0.618` (Test)
   - R²: `0.460` (Train), `0.463` (Test)

3. **Multiple Linear Regression**:
   - Lower MSE and higher R² compared to simple linear regression.
   - Final model equation: `Y = 0.8167 * MedInc + 0.1775 * HouseAge - 0.1303 * AveRooms - 0.4503 * Longitude + 2.0709`

4. **Locally Weighted Linear Regression (LWLR)**:
   - Tau values were tuned for optimal performance.
   - Tau=0.1 provided the best fit with Train MSE=`0.6765` and Test MSE=`0.7042`.

### Conclusion
The multiple linear regression model offered a balanced approach with improved prediction accuracy. Further enhancements can be achieved by exploring more complex models and feature engineering.

---

## How to Use

To explore the projects in this repository, clone the repository and navigate to the respective project directories. Each project folder contains the detailed report and the code used for analysis.

```bash
git clone https://github.com/your-username/ml-projects.git
cd ml-projects
