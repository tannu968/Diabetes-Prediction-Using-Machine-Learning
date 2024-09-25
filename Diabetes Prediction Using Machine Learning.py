import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# Load the dataset
df = pd.read_csv("E:/diabetes.csv")

# Display the dataset
print(df)

# Plot density of features
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
plt.show()

# Check for missing values
print(df.isnull().sum())

# Describe the dataset
print(df.describe())

# Display outcome distribution
print(df.Outcome.value_counts())

# Prepare the features and labels
x = df.drop('Outcome', axis=1)
Le = LabelEncoder()
df['Outcome'] = Le.fit_transform(df['Outcome'])
y = df.Outcome

# Standardize the features
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, stratify=y, random_state=10)

# Decision Tree Classifier with cross-validation
score = cross_val_score(DecisionTreeClassifier(), x, y, cv=5)
print(f'Decision Tree Cross-Validation Scores: {score}')
print(f'Mean Score: {score.mean()}')

# Bagging Classifier
bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                               n_estimators=100,
                               max_samples=0.8,
                               oob_score=True,
                               random_state=0)
bag_model.fit(x_train, y_train)
print(f'Bagging OOB Score: {bag_model.oob_score_}')
print(f'Bagging Test Score: {bag_model.score(x_test, y_test)}')

# Random Forest Classifier with cross-validation
rf_scores = cross_val_score(RandomForestClassifier(n_estimators=50), x, y, cv=5)
print(f'Random Forest Cross-Validation Scores: {rf_scores}')
print(f'Mean Score: {rf_scores.mean()}')
