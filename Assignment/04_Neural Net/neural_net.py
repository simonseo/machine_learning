import math

class NeuralNet():
	"""Implements simple neural net functions"""
	def __init__(self):
		pass
	
	def forward_propagate(self, input_layer, *weight_matrix):
		'''forward propagation algorithm from input and weight(s)'''
		
		pass

	def neuron(self, a_pre, w_intra, i):
		'''calculate an activation from previous layer and weight matrix'''
		return self.g(a_pre, w_intra[i])

	def g(self, x1, x2):
		'''logistic function of x1 and x2'''
		return 1/(1 + math.e**(-self.dot(x1, x2)))

	def dot(self, u, v):
		'''dot product of u and v'''
		mul = lambda x, y: x * y
		return sum(map(mul, u, v))

	def csv_to_matrix(self, filename):
		'''return matrix saved in csv'''
		file = open(filename, "r")
		matrix = []
		for line in file:
			matrix.append([float(el) for el in line.strip().split(",")])
		file.close()
		return matrix

def main():
	nn = NeuralNet()
	# print(nn.csv_to_matrix("ps5_theta2.csv"))

if __name__ == '__main__':
	main()