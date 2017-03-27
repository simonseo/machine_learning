import math

class NeuralNet():
	"""Implements simple neural net functions"""
	def __init__(self):
		pass

	def csv_to_list(self, filename):
		'''return 1D list saved in csv'''
		file = open(filename, "r")
		list = []
		for line in file:
			list.append(float(line.strip()))
		file.close()
		return list

	def csv_to_matrix(self, filename):
		'''return 2D matrix saved in csv'''
		file = open(filename, "r")
		matrix = []
		for line in file:
			matrix.append([float(el) for el in line.strip().split(",")])
		file.close()
		return matrix

	def max(self, array_like):
		'''simple max function for an array like structure'''
		max_i = -1
		max_val = -math.inf
		for i in range(len(array_like)): 
			el = array_like[i]
			if el > max_val:
				max_val = el
				max_i = i
		return max_i, max

	def dot(self, u, v):
		'''dot product of u and v'''
		mul = lambda x, y: x * y
		return sum(map(mul, u, v))

	def g(self, x1, x2):
		'''logistic function of x1 and x2'''
		z = -self.dot(x1, x2)
		return 1/(1 + math.e**(z))

	def neuron(self, a_pre, w_intra, i):
		'''calculate an activation from previous layer and weight matrix'''
		return self.g(a_pre, w_intra[i])
	
	def forward_propagate(self, input_layer, *weight_matrix_args):
		'''forward propagation algorithm from input and weight(s)'''
		a = input_layer
		for weight_matrix in weight_matrix_args:
			J_post = len(weight_matrix) # length of next layer
			a = [1] + a # add bias
			a = [self.neuron(a, weight_matrix, j) for j in range(J_post)]
		output_layer = a
		return output_layer

	def classifier_idx(self, input_layer, *weight_matrix_args):
		'''classifies input data as {index of maximum value in output layer}'''
		output_layer = self.forward_propagate(input_layer, *weight_matrix_args)
		max_i, max_val = self.max(output_layer)
		return max_i

	def error(self, data, label, classifier, *weight_matrix_args):
		'''calculates error (decimal) for given data, label, classifier, and weights'''
		error_count = 0
		m = len(data)
		for i in range(m):
			if label[i] != classifier(data[i], *weight_matrix_args): error_count += 1
		return error_count/m


def main():
	nn = NeuralNet()

	data = nn.csv_to_matrix("ps5_data.csv")
	w1 = nn.csv_to_matrix("ps5_theta1.csv")
	w2 = nn.csv_to_matrix("ps5_theta2.csv")
	label = nn.csv_to_list("ps5_data-labels_edit.csv")
	# print("error rate:", nn.error(data, label, nn.classifier_idx, w1, w2)) # error rate = 2.48%



if __name__ == '__main__':
	main()
