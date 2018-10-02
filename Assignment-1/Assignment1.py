import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData(file):
    data = pd.read_csv(file, sep = ",", header=None)
    return data

def createX(data):
	"""Constructs the X matrix for our data based on the dataframe
	provided
	"""
	X = []
	for index, row in data.iterrows():
		# We always consider x0 to be 1
		new_row = [1]
		
		for column in range(len(row) - 1):
			new_row.append(row[column])
		X.append(new_row)
	return np.asmatrix(X)


def scatterPlot2D(data, m , b):
	"""Takes in data, a slope, and a y intercept, and displays a scatterplot
	with a predicted regression line
	"""
	dataframe = pd.DataFrame(data)
	dfp = dataframe.plot(x=0, y=1, kind='scatter')
	dfp.set_xlabel("Population")
	dfp.set_ylabel("Profit")
	
	# For the time being, the start values are hard coded
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

def gradientDescent(theta, X, y, alpha, iterations):
	"""Takes an intial hypothesis theta, a set of features X, and a result
	vector y and performs gradient descent to find a optimal theta
	"""
	# The number of iterations we'll be using
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
	
	return np.asmatrix(new_matrix), maxes

def scaledGradientDescent(theta, X, y, alpha, iterations):
	"""Scales each parameter before running it through gradient descent.
	The fifth parameter is iterations
	"""
	scaledX, maxes = featureScale(X)
	scaled_theta =  gradientDescent(theta, scaledX, y, alpha, iterations)

	return_theta = []
	for i in range(len(maxes)):
		return_theta.append([scaled_theta.item(i) / maxes[i]])
	
	return np.asmatrix(return_theta)

		
if __name__ == "__main__":
	data = loadData("data/ex1data1.txt")
	X = createX(data)
	y = vectorize(data[1])	

	# The initial learning rate and initial paremeters (for )
	alpha = 0.01
	theta = np.matrix([[0], [1]])

	"""
	perfect_theta = normalEquation(X, y)
	new_theta = gradientDescent(theta, X, y, alpha, 10000)
	scaled_new_theta = scaledGradientDescent(theta, X, y, alpha, 10000)


	print("Perfect Theta: ", perfect_theta)
	print("Gradient Descent: ", new_theta)
	print("Scaled Gradient Descent: ", scaled_new_theta)

	scatterPlot2D(data, theta.item(1), theta.item(0))
	scatterPlot2D(data, new_theta.item(1), new_theta.item(0))
	"""
	
	print("Now doing multivariate")

	# The initial learning rate and initial paremeters (for )
	alpha = 0.01
	theta = np.matrix([[0], [0], [0]])

	data = loadData("data/ex1data2.txt")
	X = createX(data)
	y = vectorize(data[2])
	print("Original Cost:", computeCost(theta, X, y))

	scaled_new_theta = scaledGradientDescent(theta, X, y, alpha, 100000)
	perfect_theta = normalEquation(X, y)

	# Interesting result to note here
	# When gradient descent has 10000 iterations, the z value swings wildly
	# in the wrong direction, but with 100000 iterations it seems to pull itself
	# back toward the optimal value
	print("Gradient Descent: ", scaled_new_theta)
	print("Normal Equation ", perfect_theta)
	print("New Cost:", computeCost(scaled_new_theta, X, y))

	