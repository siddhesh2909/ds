import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Fill missing 'age' values with mean
df['age'] = df['age'].fillna(df['age'].mean())

# A. Distribution Plots
# a. Dist-Plot (Histogram with KDE)
plt.figure(figsize=(8, 6))
sns.histplot(df['fare'], kde=True)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# b. Joint Plot
sns.jointplot(x='age', y='fare', data=df, kind='scatter')
plt.suptitle('Age vs. Fare', y=1.02)
plt.show()

# c. Rug Plot
plt.figure(figsize=(8, 6))
sns.rugplot(df['fare'])
plt.title('Rug Plot of Fare')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.show()

# B. Categorical Plots
# a. Bar Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='pclass', y='fare', data=df)
plt.title('Average Fare by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Average Fare')
plt.show()

# b. Count Plot
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# c. Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=df)
plt.title('Age Distribution by Sex and Survival')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()

# d. Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='sex', y='age', hue='survived', data=df, split=True)
plt.title('Age Distribution by Sex and Survival')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()

# C. Advanced Plots
# a. Strip Plot
plt.figure(figsize=(8, 6))
sns.stripplot(x='sex', y='age', data=df, jitter=True)
plt.title('Age by Sex (Strip Plot)')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()

# b. Swarm Plot
plt.figure(figsize=(8, 6))
sns.swarmplot(x='sex', y='age', data=df)
plt.title('Age by Sex (Swarm Plot)')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()

# D. Matrix Plots
# a. Heat Map
group = df.groupby(['pclass', 'survived']).size().unstack()
plt.figure(figsize=(8, 6))
sns.heatmap(group, annot=True, fmt='d', cmap='Blues')
plt.title('Survival Counts by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Passenger Class')
plt.show()

# b. Cluster Map
sns.clustermap(group, annot=True, fmt='d', cmap='Blues')
plt.title('Clustered Survival Counts by Passenger Class')
plt.show()
