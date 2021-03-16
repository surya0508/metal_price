import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

cm=pd.read_csv('cm.csv')
cm.head()

cm=cm.dropna()

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
cm['commodity']=Le.fit_transform(cm['commodity'])
cm
cm['S_3'] = cm['Close'].rolling(window=3).mean()
cm['S_9'] = cm['Close'].rolling(window=9).mean()
cm['next_day_price'] = cm['Close'].shift(-1)
cm = cm.dropna()
cm['Date'] = pd.to_datetime(cm['Date'])
cm['year'] = cm['Date'].dt.year
cm['month'] = cm['Date'].dt.month
cm['day'] =cm['Date'].dt.day
X = cm[['commodity','S_3', 'S_9','year','month','day']]
y = cm['next_day_price']

from sklearn import linear_model
reg = linear_model.LinearRegression()

reg.fit(X,y)

#saving model to disk

pickle.dump(reg, open('model2.pkl','wb'))

#loading model to compare results

model=pickle.load(open('model2.pkl','rb'))
print(model.predict([[1,1732,1743,2021,6,3]]))
