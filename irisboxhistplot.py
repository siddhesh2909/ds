import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris.csv')
df

df.dtypes

df.isnull().sum()

df.hist()
plt.show()

sns.histplot(df['sepal_length'])

sns.histplot(df['sepal_width'])

sns.histplot(df['petal_length'])

sns.histplot(df['petal_width'])

df.boxplot()
plt.show()

sns.boxplot(x = 'sepal_width' , data = df)

Q1 = df['sepal_width'].quantile(0.25)
Q3 = df['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['sepal_width'] < (Q1 - 1.5 * IQR)) | (df['sepal_width'] > (Q3 + 1.5 * IQR))]

outliers

Q1 = df['sepal_width'].quantile(0.25)
Q3 = df['sepal_width'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[~((df['sepal_width'] < (Q1 - 1.5 * IQR)) | (df['sepal_width'] > (Q3 + 1.5 * IQR)))]

outliers

sns.boxplot(x = 'sepal_width' , data = outliers)


