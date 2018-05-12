import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')  # dataframe

# Changing the features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL', 'change', 'Adj. Volume']]

forecast_col = 'Adj. Close'  # Variable because we might want to forecast something else in future

# Often missing data. So replace it with dummy value and tell algo to treat it as an outlier
df.fillna(-9999, inplace=True)  # Drops the empty rows

forecast_out = int(math.ceil(0.01 * len(df)))  # Number of days we want to predict (10% later of input data)

df['label'] = df[forecast_col].shift(-forecast_out)  # shifts the rows up or down
df.dropna(inplace=True),

x = np.array(df.drop(['label'], 1))  # "x" or features = everything but the label
y = np.array(df['label'])  # "y" or label = what you are interested in
x = preprocessing.scale(x)

# Split the data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

clf = LinearRegression()  # Classifier
clf.fit(x_train, y_train)  # fit the x-y curve.
accuracy = clf.score(x_test, y_test)  # Test data!

print(accuracy)


X = np.array(df.drop(['label'], 1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out:]


