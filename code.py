#import packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Data set downloaded from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
# load in data
heart = pd.read_csv("./data/heart.csv")
print(f"Dataset dimensions: {heart.shape}")
print(heart.head())

# define X and y
X = heart.drop('HeartDisease', axis=1) #every column but HeartDisease
y = heart['HeartDisease']

# split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

#### Data visualisation ####

# draw heatmap to see correlations between numeric variables
numeric_data = heart.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# draw histograms of continuous numeric varibles
hist_data = heart[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
for variable in hist_data.columns:
    ax = sns.histplot(numeric_data[variable], edgecolor='white')
    ax.set_title(f"Histogram of {variable}")
    plt.show()

##### Data preprocessing ####

# create a selector to pick out numerical and categorical columns 
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

# apply selector to data columns
numerical_columns = numerical_columns_selector(X)
categorical_columns = categorical_columns_selector(X)

# define preprocessors for each type of column
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

#combine into one preprocessor
preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)
#### Logistic regression ####
model0 = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=1))
model0

# fit the model to the training data
model0.fit(X_train, y_train)

# evaluate the logistic regression model
y_pred = model0.predict(X_test)
print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Heart Disease']))

# cross validation
cv_scores = cross_val_score(model0, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# confustion matrix to visualise model accuracy
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#### Random forest model ####
model1= make_pipeline(preprocessor, RandomForestClassifier(n_estimators=1000, random_state=1))

# fit the model to the training data
model1.fit(X_train, y_train)

# evaluate the random forest model
y_pred = model1.predict(X_test)
print("\nRandom Forest results:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Heart Disease']))

# cross validation
cv_scores = cross_val_score(model1, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# confustion matrix to visualise model accuracy
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#### K nearest neighbors model ####
model2= make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=9))

# fit the model to the training data
model2.fit(X_train, y_train)

# evaluate the random forest model
y_pred = model2.predict(X_test)
print("\nK nearest neighbors results:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Heart Disease']))

# cross validation
cv_scores = cross_val_score(model2, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# confustion matrix to visualise model accuracy
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()