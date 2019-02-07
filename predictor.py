__author__ = 'Arjun Gurung'
__date__ = '02/06/2019'
__description__ = 'Predicting the close amount of a stock using Linear Regression ' \
                  'We use the open amount and volume to predict the close amount of the day' \
                  '(Note: not the best method for the task)'

'''
Importing all the necessary libraries and packages
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''
Collecting and analyzing the data.
'''
# Reading the csv file
df = pd.read_csv('HistoricalQuotes.csv')

'''
Viewing the data.

print(df.head())
print(df.describe())
print(df.info())
'''
print(df.head(10))

print(df.info())

df.drop(df.index[[0, 1]], inplace=True)

'''
Analyzing the data

# plot
sns.set()
plt.figure(figsize=(16, 8))
sns.lineplot(x='date', y='close', data=df)
plt.xticks(rotation=90)

sns.lineplot(x='date', y='open', data=df)
plt.xticks(rotation=90)

# Shows the plot in a different window.
plt.show()
'''

'''
Training the model
'''

'''Define X and y'''
X = df[['open', 'volume']]
y = df['close']

# Splitting the train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()

# Training the model
lm.fit(X_train, y_train)

# Let the model predict the using the training data
predictions = lm.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# Check the predictions with the testing data
sns.set()
sns.lineplot(data=y_test, palette="tab10", linewidth=2.5)

print(y_test)
print(predictions)
plt.show()
