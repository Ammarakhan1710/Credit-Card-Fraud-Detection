# Credit Card Fraud Detection using Machine Learning

This project uses a Random Forest Classifier to detect fraudulent credit card transactions. The dataset contains transactions made by European cardholders in September 2013.

## Project Overview
The goal of this project is to identify fraudulent transactions from a highly imbalanced dataset. Out of 284,807 transactions, only 492 are frauds.

## Features
Data Source Kaggle (Credit Card Fraud Detection Dataset)
Libraries Used like Pandas, Scikit-learn, Seaborn, Matplotlib, Kagglehub
Model used Random Forest Classifier
Preprocessing  used Handled missing values (0 null found) and performed 80/20 train-test split

## Results
The model achieved high precision for fraud detection, as shown in the evaluation report:
Accuracy is ~99.95%
Fraud Precision is 0.97 (Very low false alarms)
Fraud Recall is 0.72 (Detected 72% of all actual frauds)

### Confusion Matrix Insights:
Based on the testing set of 56,962 transactions:
True Negatives are 56,862
True Positives (Frauds Caught) are 71
False Alarms is 2
<img width="941" height="475" alt="image" src="https://github.com/user-attachments/assets/2700237b-f15f-4e51-bcd5-b6729cbdb15e" />


## How to Run
1. Clone this repository.
2. Install the required dependencies
   ```bash
   pip install pandas scikit-learn matplotlib seaborn kagglehub
