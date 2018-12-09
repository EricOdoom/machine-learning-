import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
breast = pd.read_csv('breast-cancer.csv')
breast.drop(['id', 'Unnamed: 32'],axis=1, inplace=True)

state = pd.get_dummies(breast['diagnosis'], drop_first=True)
breast = pd.concat([breast,state],axis=1)
breast.drop(['diagnosis'], axis=1, inplace=True)
#print(breast.columns)
X = breast.drop('M',axis=1)

y = breast['M']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=101)
log = LogisticRegression()
log.fit(X_train,y_train)
prediction = log.predict(X_test)
print(classification_report(y_test, prediction))
