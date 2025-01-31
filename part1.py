import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Load the training data
df = pd.read_csv('shoe-size-samples.csv')
df = df.dropna()

# Replace values in the 'height' column
df['height'] = df['height'].replace('1.84', '184.0')
df['height'] = df['height'].replace('1.63', '163.0')
df['height'] = df['height'].replace('1.68', '168.0')
df['height'] = df['height'].replace('1.73', '173.0')

# Convert categorical variables to numerical values
df = pd.get_dummies(df)

# Define features and target
X_pred = df.drop(columns=['sex_man', 'sex_woman'])  # Assuming 'sex' column is converted to 'sex_man' and 'sex_woman'
y_pred = df['sex_man']  # Assuming 'sex_man' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pred, y_pred, test_size=0.25, random_state=1776)

# Train the model
model = GradientBoostingClassifier(random_state=46, n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred_test = model.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_test))

# Load the new data
new_df = pd.read_csv('class_shoe_sizes_v2_2024_1.csv')
new_df = new_df.dropna()

# Replace values in the 'height' column
new_df['height'] = new_df['height'].replace('1.84', '184.0')
new_df['height'] = new_df['height'].replace('1.63', '163.0')
new_df['height'] = new_df['height'].replace('1.68', '168.0')
new_df['height'] = new_df['height'].replace('1.73', '173.0')

# Convert categorical variables to numerical values
new_df = pd.get_dummies(new_df)

# Ensure the new data has the same columns as the training data
missing_cols = set(X_pred.columns) - set(new_df.columns)
for col in missing_cols:
    new_df[col] = 0
new_df = new_df[X_pred.columns]

# Make predictions on the new data
new_predictions = model.predict(new_df)

# Add predictions to the new data
new_df['predictions'] = new_predictions

# Save the new data with predictions to a CSV file
new_df.to_csv('predicted_class_shoe_sizes_v2_2024_1.csv', index=False)

# Display the new data with predictions
print(new_df)