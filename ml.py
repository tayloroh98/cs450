import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the data
campaign = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
test = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test_mini.csv')

# Replace 'unknown' with NaN and drop missing values
campaign.replace(to_replace='unknown', value=np.nan, inplace=True)
cleaned = campaign.dropna()

# Encode features and target
features = ['job', 'default', 'loan', 'contact', 'age', 'emp.var.rate', 'day_of_week', 'cons.price.idx', 'cons.conf.idx', 'nr.employed']
X = pd.get_dummies(cleaned[features], drop_first=True)  # Use cleaned data to avoid NaN issues
y = cleaned['y']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30, random_state=42)

# Define the DecisionTreeClassifier with GridSearch for hyperparameter tuning
param_grid = {
    'max_depth': [5, 8, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
clf = DecisionTreeClassifier(class_weight="balanced", random_state=42)
grid_search = GridSearchCV(clf, param_grid, scoring='f1', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_clf = grid_search.best_estimator_

# Test the model and get the test score
test_score = best_clf.score(X_test, y_test)
print(f"Test Accuracy: {test_score}")

# Make predictions on the test set
y_pred = best_clf.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, pos_label='yes')  # Replace 'yes' with your positive class if different
print(f"F1 Score: {f1}")

# Generate the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Make predictions on the "test" database
test_features = pd.get_dummies(test[features], drop_first=True).reindex(columns=X.columns, fill_value=0)
test_predictions = best_clf.predict(test_features)
test_predictions_binary = [1 if pred == 'yes' else 0 for pred in test_predictions]

# Export predictions to a CSV file
predictions_df = pd.DataFrame({'predictions': test_predictions_binary})
predictions_df.to_csv("team2-module2-predictions2.csv", index=False)
print("Binary predictions for test database exported to team2-module2-predictions.csv")
