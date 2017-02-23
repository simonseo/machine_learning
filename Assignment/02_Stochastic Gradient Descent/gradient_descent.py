import numpy as np
import math

class GradientDescent():
	"""implementation of gradient descent algorithm for finding multi-variable non-linear regression"""
	def __init__(self, filename):
		self.filename = filename
		self.mean_stddev_list = [] # list that saves {mean, stddev} dictionaries for each variable
	
	def mean_stddev(self, data):
		"""calculates mean and standard deviation."""
		m = len(data)
		mean = sum(data)/m
		stddev = np.sqrt(sum([(x-mean)**2 for x in data])/m)
		return mean, stddev

	def normalize(self, data, filename="normalized.txt"):
		"""centers and scales the given data and saves it as filename"""
		data.seek(0)

		# read in data
		feature_matrix = [] # arrays of xij
		label_array = [] # array of y
		for line in data:
			line = [int(x) for x in line.strip().split(',')]
			# label_array.append(str(line.pop())) # separate y: we don't want to normalize y
			feature_matrix.append(line)
		feature_matrix = np.asarray(feature_matrix).T.tolist() # transpose

		# normalize data
		mean_stddev_list = [] # contains mean, stddev pair for each variable
		normal_matrix = [] # array of normalized data
		for feature_array in feature_matrix:
			mean, stddev = self.mean_stddev(feature_array)
			mean_stddev_list.append({"mean": mean, "stddev": stddev})
			normal_matrix.append([str((x - mean)/stddev) for x in feature_array])
		self.mean_stddev_list = mean_stddev_list

		# normal_matrix.append(label_array) # reappend y
		normal_matrix = np.asarray(normal_matrix).T.tolist() # transpose

		# write data
		output = open(filename, 'w')
		for line in normal_matrix:
			output.write(','.join(line) + '\n')
		output.close()

		return filename

	def gradient_descent(self, data, learning_rate=0.1, steps=100):
		"""calculates w that minimizes error using gradient descent algorithm"""
		data.seek(0)
		normal_matrix = []
		for line in data:
			line = [float(x) for x in line.strip().split(',')]
			normal_matrix.append(line)

		m = len(normal_matrix) # number of training examples = 47
		n = len(normal_matrix[0]) # number of parameters = 3
		w = [0 for i in range(n)] # Initialize w
		print("w", w, "m", m, "n", n, "alpha", learning_rate)
		for s in range(steps):
			# update w
			# if s % 10 == 0: print("step", s, ",", w[0], ",", w[1], ",", w[2])
			temp_w = w + [-1] # [w0, w1, w2, -1]
			for j in range(n):
				sigma = 0
				for i in range(m):
					x = [1] + normal_matrix[i] # [1, x1, x2, y]
					sigma += np.dot(x, temp_w) * x[j]
				temp_w[j] -= (learning_rate / m) * sigma
			w = temp_w[:-1]

			# check j(w)
			jw = 0
			for i in range(m):
				x = [1] + normal_matrix[i] # [1, x1, x2, y]
				jw += np.dot(x, w + [-1]) ** 2
			jw /= m
			if s % 10 == 9: print("step", s+1, ",", jw)



		return w

def main():
	# Create a gradient descent object
	gd = GradientDescent("housing.txt")

	# Test mean_stddev function
	mean, stddev = gd.mean_stddev(np.array([1600,2400,1416,3000]))
	result = "Passed" if (mean == 2104 and math.floor(stddev) == 635) else "Failed"
	print("mean_stddev Test", result)

	# Normalize data
	data = open(gd.filename, 'r')
	normalized_file = gd.normalize(data)
	data.close()
	data = open(normalized_file, 'r')
	w = gd.gradient_descent(data, 0.3, 80)
	print("final w:", w)
	data.close()





if __name__ == "__main__":
	main()

