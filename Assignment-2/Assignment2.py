import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class LogisticRegression(object):

	# Loads the initial parameters for the
	def __init__(self, alpha, lamda, iterations, degrees):
		self.alpha = alpha
		self.lamda = lamda
		self.iterations = iterations
		self.degrees = degrees

	def loadData(self, file):
		self.data = pd.read_csv(file, sep = ",", header=None)
		listX = []
		listY = []
		for index, row in self.data.iterrows():
			tempRow = [1]
			for i in range(len(row) - 1):
				# Loop for applying all the extra degrees of the training features
				for n in range(self.degrees):
					tempRow.append(row[i]**(n+1))
			listX.append(tempRow)
			listY.append([row[len(row) - 1]])
		
		# Turns those arrays into numpy matrixes
		self.X = np.asmatrix(listX)
		self.y = np.asmatrix(listY)
		self.m = len(self.y)
	
	def fit(self, file):
		self.loadData(file)

		# Construct the initial theta parameters
		theta = []
		for i in range(self.X.shape[1]):
			theta.append([0])

		self.theta = np.asmatrix(theta)
		self.hypothesis()
		self.modelCost = self.cost()
		print("Before fit:", self.modelCost)

		# Gradient Descent 
		for iteration in range(self.iterations):
			self.hypothesis()

			# Calculates the gradient (Without regularization)
			grad = np.dot(self.X.transpose(), (self.trainingHypothesis - self.y)) / self.m

			"""
			# By convention, theta-0 is not regularized
			regularization = [[0]]
			for j in range(1, len(theta)):
				regularization.append([(self.lamda / self.m) * self.theta.item(j)])
			regularization = np.asmatrix(regularization)

			# Completes the gradient
			grad = grad + regularization
			"""

			# Updates theta
			self.theta = self.theta - (grad * self.alpha)

		self.hypothesis()
		print("After fit:", self.cost())
		print(self.theta)


	def hypothesis(self):
		z = np.dot(self.X, self.theta)
		self.trainingHypothesis =  self.sigmoid(z)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def cost(self):
		self.trainingHypothesis = self.sigmoid(self.X.dot(self.theta))

		J = -1*(1/self.m)*(np.log(self.trainingHypothesis).T.dot(self.y)+np.log(1-self.trainingHypothesis).T.dot(1-self.y))

		if np.isnan(J[0]):
			return(np.inf)
		return(J[0])

	def predict(self, testX):
		threshold = 0.5
		p = self.sigmoid(testX.dot(self.theta)) >= threshold
		return p.astype(int)
	
	def trainingAccuracy(self):
		correct = 0
		for i in range(len(self.trainingHypothesis)):
			if (self.trainingHypothesis.item(i) >= 0.5 and self.y.item(i) == 1):
				correct += 1
			if (self.trainingHypothesis.item(i) < 0.5 and self.y.item(i) == 0):
				correct += 0
		print("Final Accuracy: ", correct / self.m)



if __name__ == "__main__":
	myModel = LogisticRegression(0.001, 1, 1000000, 1)
	#x,y,c = np.loadtxt('data/ex2data1.txt',delimiter=',', unpack=True)
	#plt.scatter(x,y,c=c)
	#plt.show()
	myModel.fit("data/ex2data1.txt")
	myModel.trainingAccuracy()
	#print(myModel.trainingHypothesis)
	#print(np.subtract(myModel.trainingHypothesis, myModel.y))
	#print(myModel.X)
