# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 2: Load the dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Display the first few rows
df.head()

# Step 3: Check data structure
df.info()

# Optional: Check for null values
df.isnull().sum()


# Step 4: Select relevant features
# Assuming the dataset has 'Age', 'EstimatedSalary' and 'Purchased'
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']


# Step 5: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Step 7: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Confusion Matrix and metrics
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()  # Unpacking confusion matrix

# Print confusion matrix
print("Confusion Matrix:\n", cm)
print(f"\nTrue Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")

# Step 10: Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nAccuracy     : {accuracy:.2f}")
print(f"Error Rate   : {error_rate:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")

