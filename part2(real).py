import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the training data
df = pd.read_csv('shoe-size-samples.csv')

# Drop rows with missing values
df = df.dropna()

# Filter out rows where sex is not 'woman' and shoe_size is less than 70
df = df[df['sex'] == 'woman']
df = df[df['shoe_size'] < 70]

# Function to convert height from feet and inches to centimeters
def convert_height_to_cm(height):
    if isinstance(height, str) and '.' in height:
        feet, inches = height.split('.')
        total_inches = int(feet) * 12 + int(inches)
    else:
        total_inches = int(height) * 12
    return total_inches * 2.54

# Apply the conversion function to the height column
df['height'] = df['height'].astype(str).apply(convert_height_to_cm)

# Define features and target
X = df[['height']]
y = df['shoe_size']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred_test = model.predict(X_test)

# Print mean squared error
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_test))

# Load the new data
df2 = pd.DataFrame({
    "#": [3, 4, 5, 8, 9, 10, 12, 14, 16, 17, 13, 24, 15, 22, 18],
    "Name": [
        "Erith H", "Kiana P", "Cydney B", "Kayla F", "Heidi N", "Danika B", "Montana D", "Jordyn S",
        "Ali C", "Keelan M", "Makalei W", "Sadie G", "Sara G", "Tyra B", "Ceirra D"
    ],
    "Yr": ["Fr.", "So.", "Fr.", "Fr.", "Fr.", "So.", "Fr.", "Fr.", "So.", "Fr.", "Fr.", "So.", "Fr.", "Fr.", "Fr."],
    "Height": ["6.0", "5.2", "5.9", "5.10", "5.8", "6.2", "5.11", "5.6", "5.5", "5.9", "5.7", "5.11", "5.10", "6.0", "5.4"],
    "Position": [
        "MB", "L", "OH", "OP/MB", "S", "MB", "OH", "DS/L/OH",
        "DS", "OH", "OH", "OH", "OH", "MB", "DS"
    ],
    "School": [
        "Ririe", "Lynnwood", "Valley View", "Lake City", "Thurston", "West Anchorage", "King Kekaulike",
        "Nathan Hale", "Marysville-Pilchuck", "Skyview", "Keaau", "Uintah", "Bishop Kelly", "Mt. Edgecumbe", "Oak Harbor"
    ],
    "Hometown": [
        "Idaho Falls, ID", "Lynnwood, WA", "Moreno Valley, CA", "Hayden, ID", "Springfield, OR",
        "Anchorage, AK", "Pukalani, HI", "Mountlake Terrace, WA", "Lake Stevens, WA", "Vancouver, WA",
        "Hilo, HI", "Vernal, UT", "Boise, ID", "Sitka, AK", "Oak Harbor, WA"
    ]
})

# Apply the conversion function to the Height column in df2
df2['height'] = df2['Height'].astype(str).apply(convert_height_to_cm)

# Predict foot size for each woman in df2
df2['Predicted_Foot_Size'] = model.predict(df2[['height']])

# Display the DataFrame with the new column
print(df2)

df2.to_csv('womanpredictedshoe.csv', index=False)