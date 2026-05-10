# ClutchPredictor

### Predicting the Most Dependable NBA Clutch Shooters Using Machine Learning

## Overview

ClutchPredictor is a machine learning project designed to answer one of basketball’s most compelling questions:

> **Which NBA players are the most dependable in clutch situations?**

Using real shot-level data from the NBA API, this project analyzes player performance during high-pressure moments and builds predictive models to determine the likelihood of a shot being made or missed.

The project compares two NBA seasons:

* **2021–2022**
* **2023–2024**

This allows for deeper analysis of:

* Changes in league shot selection
* Evolution of offensive strategies
* Clutch efficiency trends
* Model predictability across seasons

---

# Problem Statement

Traditional statistics like FG% or PPG fail to fully capture true clutch reliability because they ignore:

* Shot difficulty
* Time pressure
* Shot distance
* Player-specific context
* Shot type

This project reframes clutch performance as a **supervised machine learning classification problem**:

## Goal:

Predict whether a clutch shot will be made or missed based on contextual and player-specific features.

---

# Key Features

## Data Collection

* NBA API integration
* Multi-season shot chart data
* 29 elite NBA players
* Real game-level shot context

## Exploratory Data Analysis

* Missing value analysis
* Feature distributions
* Correlation matrices
* Class imbalance analysis
* PCA visualization
* t-SNE dimensionality reduction

## Feature Engineering

* Pressure Score
* Shot Difficulty
* Historical Player Clutch FG%
* Polynomial interaction features
* Standardization
* Feature selection

## Machine Learning Models

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)

## Hyperparameter Optimization

* GridSearchCV
* Cross-validation
* Performance benchmarking

---

# Repository Structure

```bash
cs4375_ClutchPredictor/
│
├── data/
│   ├── raw/
│   │   └── shot_data.csv
│   └── processed/
│       ├── clutch_shot_data.csv
│       └── engineered_clutch_data.csv
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   ├── data_collection.py - unused
│   ├── preprocessing.py - unused
│   ├── feature_engineering.py - unused
│   └── modeling.py - unused
│
├── reports/
│   └── ClutchPredictor_Final_Report.docx
│
├── requirements.txt
└── README.md
```

---

# Installation

## Clone the repository:

```bash
git clone https://github.com/theonlyeshann/cs4375_ClutchPredictor.git
cd cs4375_ClutchPredictor
```

---

## Create virtual environment:

### macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Main Dependencies

* nba_api
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* jupyter

---

# Workflow

---

## 1. Data Collection

Collect shot data from NBA API:

```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

### Tasks:

* Pull player shot charts
* Define clutch situations
* Save raw dataset

---

## 2. Exploratory Data Analysis

```bash
jupyter notebook notebooks/02_eda.ipynb
```

### Tasks:

* Analyze distributions
* Handle missing values
* Perform PCA / t-SNE
* Study season differences

---

## 3. Feature Engineering

```bash
jupyter notebook notebooks/03_feature_engineering.ipynb
```

### Tasks:

* Create pressure metrics
* Scale features
* Feature selection
* Encode categorical variables efficiently

---

## 4. Modeling

```bash
jupyter notebook notebooks/04_modeling.ipynb
```

### Tasks:

* Train multiple ML models
* Hyperparameter tuning
* Compare performance
* Generate confusion matrices and ROC curves

---

# Mathematical Concepts Used

---

## Logistic Regression

Used as baseline probabilistic classifier:

P(y=1|x) = 1 / (1 + e^-(β₀ + βᵀx))

---

## Decision Trees / Random Forest

* Recursive splitting
* Gini impurity
* Bagging ensembles

---

## XGBoost

Gradient boosting framework:
F_m(x) = F_(m-1)(x) + h_m(x)

---

## PCA

Dimensionality reduction:
X_pca = XW

---

## Standardization

z = (x - μ) / σ

---

# Results

### Key Findings:

* Tree-based models outperformed linear models
* Feature engineering significantly improved performance
* Pressure and player history were highly predictive
* Modern NBA seasons show different clutch shot patterns
* Random Forest / XGBoost generally performed best

---

# What Worked

* Real-world data integration
* Strong feature engineering
* Season comparison
* Multiple model benchmarking
* Advanced EDA

---

# What Didn’t

* Initial one-hot encoding caused dimensional explosion
* Neural networks risked overfitting
* Lack of defender data
* Limited contextual psychological variables

---

# Future Improvements

* Defender matchup metrics
* Spatial shot maps
* Fatigue indicators
* Real-time lineup context
* Deep learning models
* Bayesian confidence intervals

---

# Why This Project Stands Out

Unlike generic ML projects, ClutchPredictor:

* Uses live sports analytics data
* Solves a real-world strategic basketball problem
* Compares multiple seasons
* Integrates advanced feature engineering
* Applies multiple model families
* Produces interpretable player rankings

---

# Author

### Eshann Saxena

---

# License

This project is for academic and educational purposes.

---

# Final Takeaway

ClutchPredictor demonstrates how machine learning can be applied to sports analytics to evaluate player dependability under pressure, combining:

* Basketball knowledge
* Statistical rigor
* Predictive modeling
* Real-world strategic applications
