import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. DATA LOADING
# download the data in kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(f"{path}/creditcard.csv")
print("Data Loading!")

# 2. DATA PREPARATION
# Features (X) and Target (y) seperated
X = df.drop('Class', axis=1)
y = df['Class']

# Data divide Train (80%) and Test (20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data Split Complete! Training size: {len(X_train)}")

# 3. MODEL TRAINING
# Random Forest model used
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
print("Model Training Complete!")

# 4. EVALUATION
# Predictions and Accuracy check it
y_pred = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 5. VISUALIZATION
# draw a Confusion Matrix graph 
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Credit Card Fraud Detection - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()