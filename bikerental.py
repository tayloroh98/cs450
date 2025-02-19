import pandas as pd
bike = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

# Import the libraries we need 
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Commonly used modules
import numpy as np

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# import cv2
# import IPython
# from six.moves import urllib

# print(tf.__version__)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# prompt: ggplot을 이용해서 bike[casual],bike[registered]의 outlier 분포 형태를 보고싶어

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

Q1_casual = bike['casual'].quantile(0.25)
Q3_casual = bike['casual'].quantile(0.75)
IQR_casual = Q3_casual - Q1_casual

Q1_registered = bike['registered'].quantile(0.25)
Q3_registered = bike['registered'].quantile(0.75)
IQR_registered = Q3_registered - Q1_registered

# casual outlier
bike_no_outlier = bike[~((bike['casual'] < (Q1_casual - 1.5 * IQR_casual)) | (bike['casual'] > (Q3_casual + 1.5 * IQR_casual)))]

# registered outlier
bike_no_outlier = bike_no_outlier[~((bike_no_outlier['registered'] < (Q1_registered - 1.5 * IQR_registered)) | (bike_no_outlier['registered'] > (Q3_registered + 1.5 * IQR_registered)))]

bike_no_outlier['season'] = bike_no_outlier['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

bike_no_outlier = pd.get_dummies(bike_no_outlier, columns=['season'], prefix='', prefix_sep='')

bike_no_outlier.head()

bike_no_outlier['weathersit'] = bike_no_outlier['weathersit'].map({
    1: 'Clear', 
    2: 'Mist_Cloudy', 
    3: 'Light_Snow_Rain', 
    4: 'Heavy_Rain_Snow'
})

bike_no_outlier = pd.get_dummies(bike_no_outlier, columns=['weathersit'], prefix='', prefix_sep='')

bike_no_outlier.head()

# Convert 'dteday' to datetime format
bike_no_outlier['dteday'] = pd.to_datetime(bike_no_outlier['dteday'])

# Extract year and month from the date
bike_no_outlier['year'] = bike_no_outlier['dteday'].dt.year
bike_no_outlier['month'] = bike_no_outlier['dteday'].dt.month

def peak_hours(hour):
    if 6 <= hour < 10:
        return 'Morning_Peak'
    elif 10 <= hour < 16:
        return 'Afternoon'
    elif 16 <= hour < 20:
        return 'Evening_Peak'
    else:
        return 'Night'

# 새로운 열 생성
bike_no_outlier['peak_hour'] = bike_no_outlier['hr'].apply(peak_hours)

# 원-핫 인코딩 적용
bike_no_outlier = pd.get_dummies(bike_no_outlier, columns=['peak_hour'], prefix='', prefix_sep='')

bike_no_outlier['total_rentals'] = bike_no_outlier['casual'] + bike_no_outlier['registered']



X = bike_no_outlier.drop(columns=["dteday", "casual", "registered", "hr", "total_rentals"])

y = bike_no_outlier['total_rentals']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train = norm.transform(X_train)

# transform testing dataabs
X_test = norm.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Input

model = Sequential()


model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))

model.add(Dense(64))
model.add(LeakyReLU(negative_slope=0.01))

model.add(Dense(1, activation='linear'))

# opt = keras.optimizers.Adam(learning_rate=0.0001)
opt = keras.optimizers.Adam()
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=30)

history = model.fit(X_train, y_train, epochs=300, validation_split=.35, batch_size=60, callbacks=[early_stop],shuffle=False)
# history = model.fit(train_features, train_labels, epochs=2000, verbose=0, validation_split = .2, batch_size=tester2,
#                     callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)

# h = hist
hist = hist.reset_index()
# h

def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['index'], hist['mse'], label='Train Error')
    plt.plot(hist['index'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    # plt.ylim([0,50])

plot_history()

predictions = np.rint(model.predict(X_test)).astype(int)


result = root_mean_squared_error(y_test, predictions)
result

r2 = r2_score(y_test,predictions)
r2

pred = pd.DataFrame(predictions,columns=['predictions'])
pred
pred['actual'] = y_test.tolist()
pred

pred['difference'] = pred['actual']-pred['predictions']
pred

import seaborn as sns
xlims =(0,55)
ax = sns.scatterplot(data=pred,x='actual',y='predictions')
ax.plot(xlims,xlims, color='r')
plt.show()
