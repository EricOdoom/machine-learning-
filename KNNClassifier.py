#importing needed libraries
from scipy.spatial import distance

#compute euclidean distance
def eucDistance(a, b):
	return distance.euclidean(a,b)
#implmenting class
class Stark:
	#defining the fit function to remember x_ train and y_train
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	#computing the closest distance between the test point and a training point for classification
	def closestDistance(self, row, x_train):
		bestDistance = eucDistance(row, x_train[0])
		bestIndex =0

		for i in range(1, len(x_train)):
			new_distance = eucDistance(row, x_train[i])
			if new_distance < bestDistance:
				bestDistance = new_distance
				bestIndex = i 
		return self.y_test[bestIndex]

	#defining the predict function to predict our test value labels using our closest distance function
	def predict(self,x_test):
		predictions = []
		for row in x_test:
			label =  self.closestDistance(row)
			predictions.append(label)
		return predictions


