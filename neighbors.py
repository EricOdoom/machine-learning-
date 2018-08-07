#Analyzing data using the K Nearest Neighbors algorithm


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('Classified Data', index_col=0)
print(data.head())
print('\n')
print('\n')
print(data.info())
print('\n')
print('\n')
print(data.describe())
print('\n')
print('\n')
print(data.keys())

scaler = StandardScaler()
scaler.fit(data.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(data.drop('TARGET CLASS', axis=1))


x = scaled_features
print('\n')
y = data['TARGET CLASS']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

error = []
for i in range(1,500):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(x_train, y_train)
	predictions_i = knn.predict(x_test)
	error.append(np.mean(predictions_i != y_test))
print('\n')
best_k = np.argmin(error)
print('best k value is ', best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print('\n')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

plt.plot(range(1,500), error, color='cyan', linestyle = 'dashed', marker = 'o')
plt.title('Error rate versus k value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()

