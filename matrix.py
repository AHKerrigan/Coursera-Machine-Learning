import numpy as np
from numpy.linalg import inv

A = np.matrix([[1, 3],
              [4, 0],
              [2, 1]])
B = np.matrix([[1],
               [5]])

C = np.matrix([[3, 4],
               [2, 16]])
invC = inv(C)
#print(C)
#print(invC)
#print(A)
#print(B)
#print(A * B)

u = np.matrix([1, 3, -1])
v = np.matrix([[2],
               [2],
               [4]])

print(C * invC)
print(u * v)