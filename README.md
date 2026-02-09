# Comparative Evaluation of Machine Learning Models for Customer Churn Prediction

## Research Motivation
In real-world applications, model reliability and consistency are as important as predictive accuracy. This project focuses on disciplined experimental design and fair comparison of supervised learning models.

## Dataset
Telco Customer Churn dataset (Kaggle), containing customer demographics, service usage, and billing information.

## Experimental Design
- Standardized preprocessing for all models
- Same feature transformations across experiments
- No model-specific tuning to ensure fair comparison
- 5-fold cross-validation for robust evaluation

## Models Evaluated
- Logistic Regression
- Support Vector Machine
- Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

Mean and standard deviation are reported to describe performance consistency.

## Visualization
A single bar plot of mean F1-score is included to support comparison without unnecessary graphical complexity.

## Key Findings
- Model performance varies across evaluation metrics.
- More complex models may achieve higher mean performance but show higher variability.
- No single model consistently dominates across all metrics.
- Results highlight the importance of systematic evaluation rather than relying on a single metric.

## Research Contribution

This project demonstrates a reproducible machine learning pipeline with emphasis on fairness, consistency analysis, and research-oriented evaluation practices.

Model f1 comparison
<br>
<br>
<img src="https://github.com/pratimadhende/ml-model-comparative-study/blob/869a0db726c3ba7693491b3ac8867a1e5fc387a2/model_f1_comparison.png" alt="Image Discription" width="500">
