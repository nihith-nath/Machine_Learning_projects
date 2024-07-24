
# Machine Learning Projects

## Introduction
Welcome to my Machine Learning GitHub repository! This repository contains various machine learning projects that I have worked on, including clustering, classification, and regression analyses. Each project includes detailed reports and insights derived from the data.

## Table of Contents
1. [Clustering Analysis](#clustering-analysis)
2. [MNIST Dataset Classification](#mnist-dataset-classification)
3. [California Housing Price Prediction](#california-housing-price-prediction)

---

## Clustering Analysis
### Project Overview
This project focuses on clustering analysis to identify patterns and group similar data points. The Silhouette Score is used to evaluate the clustering performance.

### Results
- **Silhouette Score:** 0.75 (indicating good clustering structure)

### Key Insights
- The clustering analysis successfully grouped data points with similar characteristics.
- High silhouette score suggests well-defined clusters.

---

## MNIST Dataset Classification
### Project Overview
This project involves classifying handwritten digits from the MNIST dataset using various machine learning models.

### Data Preprocessing
- Images were normalized to facilitate model convergence.
- Data was split into training and testing sets.

### Algorithms and Results
1. **Logistic Regression**
   - **Best Regularization Parameter (C):** 0.1
   - **Test Accuracy:** 91%
   - **Precision:** High precision for class '1' (0.96)
   - **Recall:** High recall for class '0' (0.98)
   - **F1-Score:** Consistent across most classes, averaging at 0.91

### Key Insights
- Logistic Regression achieved a peak accuracy of 91% with a regularization parameter (C) of 0.1.
- The model demonstrated a good balance between precision and recall.

---
## California Housing Price Prediction
### Project Overview
This project aims to predict median house values in California using the 1990 California census dataset.

### Data Features
- **MedInc:** Median income in block group
- **HouseAge:** Median house age in block group
- **AveRooms:** Average number of rooms per household
- **AveBedrms:** Average number of bedrooms per household
- **Population:** Block group population
- **AveOccup:** Average house occupancy
- **Latitude:** Block group latitude
- **Longitude:** Block group longitude
- **MedHouseVal:** Median house value for households within a block group

### Models and Results
1. **Simple Linear Regression**
   - **Train MAE:** 0.617
   - **Test MAE:** 0.618
   - **Train MSE:** 0.716
   - **Test MSE:** 0.720
   - **Train R²:** 0.460
   - **Test R²:** 0.463

2. **Multiple Linear Regression**
   - **Train MSE:** 0.5508
   - **Test MSE:** 0.5634
   - **R-squared:** 0.5749

3. **Locally Weighted Linear Regression (LWLR)**
   - **Tau = 0.1:**
     - Train MSE: 0.6765
     - Test MSE: 0.7042
     - Train R²: 0.4974
     - Test R²: 0.4593

### Key Insights
- The Simple Linear Regression model had moderate performance with an R² score of 0.463.
- The Multiple Linear Regression model improved the prediction accuracy with an R² score of 0.5749.
- LWLR provided varying accuracy levels, with Tau=0.1 offering the best fit.

---

Feel free to explore the repository to find detailed reports and code implementations for each project. Your feedback and suggestions are welcome!

---
