from warmUpExercise import *
from plotting import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    data = pd.read_csv("data/ex1data1.txt", sep = ",", header=None)
    return data

def createX(data):
	X = []
	for feature in data[0]:
		X.append([1, feature])
	return np.asmatrix(X)


def scatterPlot(data):
    dataframe = pd.DataFrame(data)
    print(dataframe)
    dfp = dataframe.plot(x=0, y=1, kind='scatter')
    dfp.set_xlabel("Population")
    dfp.set_ylabel("Profit")
    plt.show()

def computeCost(theta, X,  y):
	"""Takes in a hypothesis matrix and the feature vector, then computes
	the current cost of the hypothesis
	"""

	sum = 0
	m = len(X)
	# Creates the matrix of predictions for the given theta
	predictions = X * theta
	for i in range(m):
		sum = sum + (predictions[i] - y[i])**2
	
	J = sum / (2 * m)
	return J

def vectorize(features):
	"""Takes in a set of features and returns a single column vector matrix
	"""
	y = []
	for item in features:
		y.append([item])
	return np.asmatrix(y)

def gradientDescent(theta, X, y, alpha):
	"""Takes an intial hypothesis theta, a set of features X, and a result
	vector y and performs gradient descent to find a optimal theta
	"""
	# The number of iterations we'll be using
	iterations = 1500

	for interation in range(iterations):
		tempTheta = []
		predictions = X * theta
		for j in len(theta):
			sum = 0
			for i in range(m):
				sum = sum + ((predictions.item() - y[i]) * x[j])


if __name__ == "__main__":
	warmUp()
	data = loadData()
	X = createX(data)
	y = vectorize(data[1])

	# The initial learning rate
	alpha = 0.01
	theta = np.matrix([[0], [0]])
