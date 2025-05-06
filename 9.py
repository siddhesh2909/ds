import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("titanic.csv")

# Check missing values and fill them using backward fill
data = data.bfill()

# Boxplot of Age by Sex and Survived
sns.boxplot(data=data, x="Sex", y="Age", hue="Survived")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# Z-score calculation for Age
mean_age = data['Age'].mean()
std_age = data['Age'].std()
data['zscore'] = (data['Age'] - mean_age) / std_age

# Detect outliers
outliers = data[np.abs(data['zscore']) > 3]
print(outliers[['Age', 'Sex', 'Survived', 'zscore']])

# Remove outliers
titanic_cleaned = data[np.abs(data['zscore']) <= 3]
titanic_cleaned = titanic_cleaned.drop(columns=['zscore'])

# Print dataset sizes
print("Original dataset size:", data.shape[0])
print("Cleaned dataset size:", titanic_cleaned.shape[0])

# Show first rows of cleaned dataset
print(titanic_cleaned.head())
