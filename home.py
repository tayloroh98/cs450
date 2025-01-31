import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('shoe-size-samples.csv')
df = df.dropna()

df['height'].replace('1.84', '184.0')
df['height'].replace('1.63', '163.0')
df['height'].replace('1.68', '168.0')
df['height'].replace('1.73', '173.0')

df = df.drop(columns=['time'])

df.to_csv('modified_shoe_size_samples.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

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


