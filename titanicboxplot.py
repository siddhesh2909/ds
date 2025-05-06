import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')
df

df.isnull().sum()

df.dtypes

sns.boxplot(x="sex", y="age", hue="survived", data=df)

df['age'] = df['age'].fillna(df['age'].mean())

sns.boxplot(x="sex", y="age", hue="survived", data=df)

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR))]

outliers

Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))]

outliers

sns.boxplot(x="sex", y="age", hue="survived", data=outliers, showfliers = False)
