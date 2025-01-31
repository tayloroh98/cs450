import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('shoe-size-samples.csv')

df = df.dropna()
df = df[df['sex'] != 'man']
df = df[df['shoe_size'] < 70]

# Convert 'height' column to float and replace values
df['height'] = df['height'].astype(float)
df['height'] = df['height'].replace(1.84, 184.0)
df['height'] = df['height'].replace(1.63, 163.0)
df['height'] = df['height'].replace(1.68, 168.0)
df['height'] = df['height'].replace(1.73, 173.0)

df = pd.DataFrame(df)

data = {
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
}

# Create a pandas DataFrame
df2 = pd.DataFrame(data)

# Display the DataFrame



def convert_height_to_cm(height):
    if '.' in height:
        feet, inches = height.split('.')
        total_inches = int(feet) * 12 + int(inches)
    else:
        total_inches = int(height) * 12
    return total_inches * 2.54

df2['Height'] = df2['Height'].apply(convert_height_to_cm)

print(df2)