#%%
import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/msans/PycharmProjects/PythonProject2/test",sep="\t")

# Identify null values per column
null_counts = df.isnull().sum()

print("Missing values per column:")
print(null_counts)

# Show only columns with missing values
print("\nColumns with missing values:")
print(null_counts[null_counts > 0])

#%%
df = pd.read_csv("C:/Users/msans/PycharmProjects/PythonProject2/test", sep=",", header=0)
df = pd.read_csv("C:/Users/msans/PycharmProjects/PythonProject2/test", sep=None, engine="python")
print(df.columns[:10])

#%%
from sklearn.preprocessing import LabelEncoder

# Clean column names
df.columns = df.columns.str.strip()

# Define ordinal columns (only keep those present in df)
ordinal_cols = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC"
]

ordinal_cols = [col for col in ordinal_cols if col in df.columns]  # keep only existing

# Apply LabelEncoder
le = LabelEncoder()
for col in ordinal_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("Encoded ordinal columns:\n", df[ordinal_cols].head())

#%%
from sklearn.preprocessing import LabelEncoder

# keep only those ordinal columns that exist in your DataFrame
ordinal_cols = [col for col in ordinal_cols if col in df.columns]

le = LabelEncoder()
for col in ordinal_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # convert NaN to string

print("Encoded ordinal columns:")
print(df[ordinal_cols].head())

#%%
import pandas as pd

# Load your dataset (adjust sep if needed)
df = pd.read_csv("C:/Users/msans/PycharmProjects/PythonProject2/test", sep=None, engine="python")

# List numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numeric columns available:", numeric_cols)

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/msans/PycharmProjects/PythonProject2/test", sep=None, engine="python")  # adjust sep if needed

# Target variable
y = df["GrLivArea"]  # you can also use LotArea, OverallQual, etc.

# Features: drop target a

#%%
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/msans/PycharmProjects/PythonProject2/test", sep=None, engine="python")  # adjust sep if needed

# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Compute skewness
skewed_feats = df[numeric_cols].skew().sort_values(ascending=False)
skewed_feats = skewed_feats[skewed_feats > 0.75]  # consider skew > 0.75 as high

print("Highly skewed numeric features:")
print(skewed_feats)
#%%
# Apply log1p to reduce skewness
for col in skewed_feats.index:
    df[col] = np.log1p(df[col])  # log1p(x) = log(1 + x)

# Check skewness after transformation
skewed_after = df[skewed_feats.index].skew()
print("\nSkewness after log1p transformation:")
print(skewed_after)

#%%
