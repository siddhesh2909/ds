import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

data['species'] = iris.target

missing_values = data.isnull().sum()
data_description = data.describe()
data_dimensions = data.shape

data_types = data.dtypes
data['species'] = data['species'].astype('category') 
data_types_after_conversion = data.dtypes

data['species'] = data['species'].cat.codes  

print("Missing Values:\n", missing_values)
print("\nInitial Statistics:\n", data_description)
print("\nDimensions of the DataFrame:", data_dimensions)
print("\nData Types Before Conversion:\n", data_types)
print("\nData Types After Conversion:\n", data_types_after_conversion)
print("\nData after Conversion:\n", data.head())
