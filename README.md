House Prices Analysis and Feature Importance
Overview

This project involves preprocessing the House Prices dataset, handling missing and categorical data, applying transformations on skewed features, encoding ordinal and nominal variables, and analyzing feature importance using a Random Forest model.

1. Dataset Loading

Loaded the dataset using Pandas.

Verified numeric and categorical columns.

Checked for missing values using:

df.isnull().sum()

2. Handling Missing Values

Identified columns with missing values.

Imputed or removed missing values depending on the feature.

Checked missing values after preprocessing.

3. Encoding Categorical Variables
a) Ordinal Features

Features like ExterQual, ExterCond, BsmtQual, etc., were ordinal.

Used LabelEncoder to convert them into numeric format:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[ordinal_col] = le.fit_transform(df[ordinal_col].astype(str))

b) Nominal Features

Used One-Hot Encoding (pd.get_dummies()) for categorical variables without order:

df = pd.get_dummies(df, drop_first=True)

4. Skewness Handling

Identified highly skewed numeric features:

skewed_feats = df[numeric_cols].skew()


Applied log transformation to reduce skewness:

import numpy as np
for col in skewed_feats.index:
    df[col] = np.log1p(df[col])


Checked skewness after transformation to confirm improvement.

5. Feature Importance using RandomForest

Selected a numeric target column (e.g., GrLivArea or OverallQual).

Split dataset into features (X) and target (y), encoding categorical variables as needed.

Trained a Random Forest Regressor:

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


Extracted feature importances:

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feat_importances.sort_values(ascending=False).head(20)


Visualized top 20 important features with a horizontal bar chart.

6. Key Notes

np.log1p() is used for log transformation as it handles zeros safely.

Label encoding of ordinal variables preserves the order (alphabetical order by default; for strict ordinal order, OrdinalEncoder can be used with categories defined manually).

One-hot encoding prevents model misinterpretation of nominal features as ordinal.

Feature importance provides insights into which features most influence the target variable.

7. Libraries Used

pandas → data handling

numpy → numerical operations

sklearn → LabelEncoder, RandomForestRegressor, train_test_split

matplotlib → plotting feature importance

8. Future Steps

Tune Random Forest hyperparameters for better predictive accuracy.

Experiment with other models like Gradient Boosting or XGBoost.

Consider cross-validation and performance metrics (MAE, RMSE, R²) for model evaluation.
