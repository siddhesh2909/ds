import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")
df.head()

grouped_stats = df.groupby('Species').agg(['mean', 'median', 'min', 'max', 'std'])
print("Summary Statistics grouped by species:\n", grouped_stats)

species_groups = df.groupby('Species')['SepalLengthCm'].apply(list)
print("\nList of numeric values for each species:")
print(species_groups.to_dict())

setosa = df[df['Species'] == 'Iris-setosa'].describe()
versicolor = df[df['Species'] == 'Iris-versicolor'].describe()
virginica = df[df['Species'] == 'Iris-virginica'].describe()

print("\nDescriptive Stats for Iris-setosa:")
display(setosa)

print("\nDescriptive Stats for Iris-versicolor:")
display(versicolor)

print("\nDescriptive Stats for Iris-virginica:")
display(virginica)
