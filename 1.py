# 1. Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Loading the dataset
df = pd.read_csv("Iris.csv")  # Ensure the file is in the same directory or provide the full path

# 3. Displaying first few rows
print(df.head())

# 4. Checking datatypes
print(df.dtypes)

# 5. Checking for null values
print(df.isnull().sum())

# 6. Visualizing count plots for each feature
sns.histplot(df["SepalLengthCm"])
plt.xlabel("SepalLengthCm")
plt.ylabel("Count")
plt.show()

sns.histplot(df["SepalWidthCm"])
plt.xlabel("SepalWidthCm")
plt.ylabel("Count")
plt.show()

sns.histplot(df["PetalLengthCm"])
plt.xlabel("PetalLengthCm")
plt.ylabel("Count")
plt.show()

sns.histplot(df["PetalWidthCm"])
plt.xlabel("PetalWidthCm")
plt.ylabel("Count")
plt.show()

# 7. Filtering rows based on SepalWidthCm > 4.0
filtered_df = df[df["SepalWidthCm"] > 4.0]
print(filtered_df)

# 8. Displaying a sample of the cleaned dataset
print(df.head())
