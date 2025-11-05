# Obesity Level Classification Dashboard

[Try online dataset and model prediction](https://ml10projectdeployment.streamlit.app/)
![Obesity Dashboard](images/dashboard_screenshot.png)

## Overview

This project is an **interactive machine learning dashboard** for predicting obesity levels based on personal attributes.  
It leverages **Random Forest classification**, **UMAP visualization**, and **publication-grade plots** to allow users to explore the dataset and predict obesity classes interactively.

The dashboard is built with **Streamlit**, making it easy to deploy and share.

---

## Dataset

The dataset is sourced from:

**“A study on the estimation of obesity levels based on eating habits and physical condition”**  
[PubMed link](https://pubmed.ncbi.nlm.nih.gov/31467953/)

**Features include:**  

- **Numerical:** Age, Height, Weight, etc.  
- **Categorical:** Gender, Family history, Frequent eating habits, etc.  
- **Target:** Obesity level (e.g., `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, ..., `Obesity_Type_III`)

---

## Project Goals (Business Case)

- Provide a **user-friendly tool** for obesity level prediction.
- Enable **interactive exploration** of relationships between features (scatter plots, heatmaps, UMAP).
- Showcase **model performance metrics**, including probabilities and ROC curves.
- Make a **deployable ML application** for portfolio and real-world use.

---

## Features

### 1. Data Exploration
- View dataset head and summary statistics.
- Understand distributions of numerical and categorical features.

### 2. Interactive Visualizations
- **Scatter Plot:** Choose any two numeric features and optionally color by a categorical feature.
- **Correlation Heatmap:** Shows correlation matrix of numeric features.
- **UMAP Projection:** Reduces high-dimensional numeric features into 2D space to visualize clusters.

### 3. Obesity Prediction
- **Input personal features** via sliders or dropdowns.
- **Predict obesity level** using a trained Random Forest model.
- **View prediction probabilities** for all classes.
- **ROC Curve:** Visualize model performance per class.

---

## Model Details

- **Algorithm:** Random Forest Classifier
- **Preprocessing:**
  - Scaling for numerical features
  - One-hot encoding for categorical features
- **Hyperparameter Tuning:** `GridSearchCV` to optimize number of trees, max depth, min samples, and max features.
- **Saved Model:** `random_forest_obesity.pkl`

**Performance Example:**
