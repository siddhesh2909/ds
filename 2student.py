import pandas as pd
import numpy as np

df=pd.read_csv("student_admission.csv")
df

df.info()

df.describe()

df.head()

df.isnull()

df.isnull().sum()

df.ffill()

df.bfill()

df.fillna(5)

df.replace(to_replace=np.nan,value=5)

df

numeric_columns=df.select_dtypes(include='number').columns
numeric_columns

for col in numeric_columns:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-(1.5*IQR)
    upper_bound=Q3+(1.5*IQR)
    print(f"{col} :Q1 : {Q1}, Q3 :{Q3}, IQR : {IQR} , outliers: Lower bound = {lower_bound}, upper bound = {upper_bound}")
    cleaned_data=df[(df[col]>=lower_bound)&(df[col]<=upper_bound)]
    print(f"Cleaned {col} data without outliers:\n{cleaned_data[col].head()}")

df

for col in numeric_columns:
    print(col)
    vicky=(df[col]-df[col].mean())/(df[col].std())
    print(vicky)


df


