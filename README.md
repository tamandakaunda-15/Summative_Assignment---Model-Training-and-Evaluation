# Predicting Student Dropout in Malawi Using Machine Learning Classification Models

## Problem Statement

Despite education being a fundamental right, Malawi faces a severe learning crisis with significant dropout rates and poor learning outcomes. Only  58.5% of children complete the first four years of primary education(UNICEF), and just 19% of children aged 7 to 14 have foundational reading skills, while 13% have numeracy skills(Malawi Human Capital Index 2020).  Girls are mostly affected, with more girls dropping out than boys across all educational levels (UNESCO Institute of Statistics, 2016). This project develops machine learning models to predict student dropout risk, to enable targeted action points that reduce the risks of further dropouts.

Dataset Linked Here; https://www.kaggle.com/datasets/edgargulay/secondary-school-student-dropout

## Dataset and Context

This project analyzes student data incorporating multi-dimensional factors critical to Malawi's educational landscape:
- **Demographics**: Age, gender, location, marital status
- **Socioeconomic factors**: Family income, parental education, poverty indicators
- **Social factors**: Early marriage risk, teenage pregnancy, community support
- **Academic factors**: Historical performance, attendance records
- **Infrastructure**: School accessibility, distance from home

## Dropout Prediction Model Implementation

The aim of this project is to predict student dropout in both primary and secondary school by analyzing the demographics, socioecomic and school-related factors. The goal is to identify patterns in student behavior that signal potential risk of dropping out. I built and compared different classification models using traditional ML algorithms and neural networks. I used the Student Dropout Dataset with over 670 records and 33 features that are related to education and lifestyle. 



## The Combinations that worked better
Model 2,  which used the Adam optimizer, L2 regularization, and a dropout of 0.3, performed best overall.  This model showed a strong accuracy and F1 score while maintaining good generalization due to the combination of both regularization and adaptive learning. The L2 (ridge) helped in controlling model complexity and reduced overfitting, especially with dropout.




 ## Neural Network  and Traditional Machine Learning comparison
Traditional machine learning (the random forest model)  was the best overall model , even outperfoming all neural networks as indicated in its raw metrics.
SVM was the best-performing traditional model with kernel = ‘rbf’ and C=1.
Neural networks performed better after applying L2 regularization, dropout, and early stopping. 
The best NN model still fell slightly short of the random forest but offered insight into how deep models can be improved with optimized tuning. 


Note: I experimented with several traditional machine learning algorithms, including SVM and logistic regression, but random forest gave the best results and is the one I used for the final comparison.


## Error Analysis
Each model was evaluated using:
\F1 Score
\Precision
\Recall
\Accuracy
\ROC-AUC
\Confusion Matrix

Matplotlib, Seaborn, and sklearn were used to visualize the evaluation.


## Summary of Results


| Instance     | Optimizer        | Regularizer | Learning Rate | Epochs | Early Stopping | Dropout | Accuracy | Loss | F1-Score | Precision | Recall | AUC-ROC |
| ------------ | ---------------- | ----------- | ------------- | ------ | -------------- | ------- | -------- | ------------------------ | -------- | --------- | ------ | ------- |
| 1 (Baseline) | Default (`Adam`) | None        | 0.001         | 50     |  No           | 0.0     | 0.9462   | 0.25                   | 0.8293   | 0.8095    | 0.85   | 0.9691  |
| 2            | Adam             | L2          | 0.001         | 100    | Yes          | 0.3     | 0.9692   | 0.18                   | 0.9000   | 0.9000    | 0.90   | 0.9809  |
| 3            | RMSprop          | L1          | 0.001         | 100    |  Yes          | 0.2     | 0.9846   | 0.12                   | 0.9474   | 1.0000    | 0.90   | 0.9918  |
| 4            | SGD              | L1\_L2      | 0.001         | 200    |  Yes          | 0.4     | 0.8462   | 0.35                   | 0.0000   | 0.0000    | 0.00   | 0.7977  |


**Note**: Loss values are estimated based on the final training and validation loss curves.

The Random Forest model was the top perfoming of all becuase it had a perfect accuracy , F1 score and an AUC.
Model 3 (RMSprop + L1 + Early Stopping) had the best performance with an F1-score of 0.9474 and an AUC of 0.9918 among the neural networks.

The SDG model significantly struggled despite using regularization and dropout—possibly due to its sensitivity to learning rate and convergence challenges.

Models with a combination of dropout, regularization, and early stopping generally performed better than the baseline.

While optimized neural networks with RMSprop and L1 regularization achieved a strong F1 score of 0.9474, the Random Forest algorithm outperformed all models with a perfect classification (F1 SCORE	1.0000). Therefore, for this dataset, Random Forest proved more effective.

## ML Hyperparameters (for Random Forest):
N-estimators: 200, max_depth: 20, min_samples_split: 2


## Model Loading and Testing
```python
import tensorflow as tf
import joblib
import numpy as np

# Load the best model (example for neural network)
model = tf.keras.models.load_model('saved_models/optimized_model_2.h5')

# Load preprocessing components
scaler = joblib.load('saved_models/scaler.pkl')

# Make predictions on new data
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
risk_scores = predictions[:, 0]  # Dropout probability

# Categorize risk levels
high_risk = risk_scores > 0.7
medium_risk = (risk_scores > 0.4) & (risk_scores &lt;= 0.7)
low_risk = risk_scores &lt;= 0.4
