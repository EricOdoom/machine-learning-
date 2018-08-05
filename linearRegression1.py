#Importing necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd 

#loading data
customers = pd.read_csv('Ecommerce Customers')

#chechking data facts
print(customers.head(3))
print(customers.info())
print(customers.describe())
print(customers.keys())

#cheking the correlation between parts of the data by visualizing 
sns.lmplot(x= 'Length of Membership',y='Yearly Amount Spent', data=customers)
sns.lmplot(x= 'Time on App',y='Yearly Amount Spent', data=customers)
sns.lmplot(x= 'Time on Website',y='Yearly Amount Spent', data=customers)
sns.lmplot(x= 'Avg. Session Length',y='Yearly Amount Spent', data=customers)

#assigning x and y to specific parts of the whole data for analysis
x = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

#training  and testing data
linear = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4)
linear.fit(x_train, y_train)
predictions = linear.predict(x_test)

#checking the data predicted by the model
print(predictions)

#checking the accuracy of the model by finding the gradient using the origin as starting point.
#The closer the values are to 1.0, the more accurate they are
for i in range(1,len(predictions)):
	print(predictions[i]/y_test[i])

#checking how much our predicted values and actual values correlate bby visualizing on a scatter plot
plt.scatter(y_test, predictions)
plt.xlabel('Actual values')
plt.ylabel('predicted values')
plt.title('Actual values versus predicted vaues')
plt.legend()
plt.show()
