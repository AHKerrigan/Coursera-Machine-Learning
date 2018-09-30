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


def scatterPlot(data, m , b):
	"""Takes in data, a slope, and a y intercept, and displays a scatterplot
	with a predicted regression line
	"""
	dataframe = pd.DataFrame(data)
	dfp = dataframe.plot(x=0, y=1, kind='scatter')
	dfp.set_xlabel("Population")
	dfp.set_ylabel("Profit")
	plt.plot([5, 25], [(5 * m + b), (25 * m + b)], 'k-', lw=2)
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
	iterations = 10000
	m = len(y)
	for interation in range(iterations):
		tempTheta = []
		predictions = X * theta

		# Performs the gradient descent algorithm on each feature 
		for j in range(len(theta)):
			sum = 0

			# The summation for the partial derivative with respect to theta-j
			# The sum of the difference between the prediction and actual value
			# times the corresponding x value
			for i in range(m):
				sum = sum + ((predictions.item(i) - y.item(i)) * X.item(i, j))
			pDev = sum / m

			# Multiplies the partial derivative by the learning rate
			# Then subtracts that value from the current theta and updates the 
			# theta vector 
			tempTheta.append([theta.item(j) - (alpha * pDev)])
		theta = np.asmatrix(tempTheta)
	
	return theta

def normalEquation(X, y):
	"""Computes the theta matrix using the normal Equation
	"""
	xTx = np.transpose(X).dot(X)
	XtX = np.linalg.inv(xTx)
	XtX_xT = XtX.dot(np.transpose(X))
	theta = XtX_xT.dot(y)
	return theta

def featureScale(features):
	"""Scales the features such that each feature percentage of the maximum value
	"""
	maxes = []
	new_matrix = []

	# Finds the maximum value in each feature by iterating through the transpose of the 
	# matrix
	for column in np.transpose(features):
		maxes.append(np.amax(column))
	
	for row in features:
		new_column = []
		for x in range(row.size):
			new_column.append(row.item(x) / maxes[x])
		new_matrix.append(new_column)
	
	return np.asmatrix(new_matrix)
		
if __name__ == "__main__":
	data = loadData()
	X = createX(data)
	y = vectorize(data[1])
	scaledX = featureScale(X)	

	# The initial learning rate
	alpha = 0.01
	theta = np.matrix([[0], [1]])

	perfect_theta = normalEquation(X, y)
	new_theta = gradientDescent(theta, X, y, alpha)
	scaled_perfect_theta = normalEquation(scaledX, y)
	scaled_new_theta = gradientDescent(theta, scaledX, y, alpha)
	
	print("Perfect Theta: ", perfect_theta)
	print("Gradient Descent: ", new_theta)
	print("Scaled Perfect Theta: ", scaled_perfect_theta)
	print("Scaled Gradient Descent: ", scaled_new_theta)


	scatterPlot(data, theta.item(1), theta.item(0))
	scatterPlot(data, new_theta.item(1), new_theta.item(0))
	
