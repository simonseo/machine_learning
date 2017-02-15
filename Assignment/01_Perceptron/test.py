import numpy as np

def pylist():
	x = []
	for i in range(100):
		temp = []
		for j in range(100):
			temp.append(j)
		x.append(temp)
	# print(x)

def numpylist():
	x = np.zeros((1, 100))
	for i in range(100):
		temp = np.array([])
		for j in range(100):
			temp = np.append(temp, j)
		x = np.append(x, [temp], 0)
	x = np.delete(x, 0, 0)
	# print(x)

