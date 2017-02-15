from operator import mul, add
from itertools import repeat
from os import path
from heapq import nlargest, nsmallest
import numpy

'''To do list
switch to numpy
implement CLI

'''

def dot_vec(A, B):
	"""dot product of two same length array that are elementwise multiplicable"""
	return sum(map(mul, A, B))

def add_vec(A, B):
	"""elementwise addition of two arrays of same length"""
	return list(map(add, A, B))

def scal_vec(n, A):
	"""scalar multiplication of A by n"""
	return list(map(mul, repeat(n), A))

def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices


class Perceptron():
	"""implementation of the Perceptron algorithm for classifying spam emails

	In this implementation, the labels provided as the first column of each email
	in the data are not counted as features.

	"""
	def __init__(self, training_data_file, threshold, debug):
		print("Perceptrons! Roll out!")
		self.threshold = threshold
		self.training_data_file = training_data_file
		self.debug = debug
		self.features = [] # list of features in consideration
		self.feature_vector_list = [] # list of feature vectors of each email
		self.label_list = [] # list of labels for each email

	def words(self, data, X):
		"""creates a list of words that occur in at least X emails"""
		data.seek(0)
		word_occurrence = {}

		for email in data:
			email = email.split()
			email.pop(0)  # remove label
			email = list(set(email)) # list of unique words
			for word in email:
				if word in word_occurrence.keys():
					word_occurrence[word] += 1
				else:
					word_occurrence[word] = 1

		features = []
		for key in word_occurrence.keys():
			if word_occurrence[key] >= X:  # any words fewer than X is ignored
				features.append(key)

		return features

	def check_words(self):
		"""checks words function"""
		print("====== Check words(data, X) start ======")
		print(self.features)
		print('length', len(self.features))
		print('httpaddr', 'httpaddr' in self.features)
		print('1', '1' in self.features)
		print('0', '0' in self.features)
		print("====== Check words(data, X) finish ======")

	def feature_vector(self, email):
		"""creates a feature vector from an email"""
		if not len(self.features) > 0: # if self.features is not defined
			raise MyLookupException({"message":"variable has not been initiated", "variable":"self.features"})
		email = email.split()
		email.pop(0)  # remove label
		feature_vector = []
		for word in self.features:
			if word in email:
				feature_vector.append(1)
			else:
				feature_vector.append(0)
		return feature_vector

	def check_feature_vector(self):
		"""checks basic functionality of feature_vector function"""
		print("====== Check feature_vector(email) start ======")
		print("feature vector for one email: ")
		print(self.test_feature_vector)
		data = open(self.training_data_file, "r")
		email = data.readline()
		print(email)
		email = email.split()
		data.close()
		test_pass = True
		for i in range(len(self.test_feature_vector)):
			el = self.test_feature_vector[i]
			if el == 1:
				if not (self.features[i] in email):
					print("feature vector purity test failed")
					test_pass = False
					break
				else:
					print(self.features[i])
		if test_pass: print("feature vector purity test passed")
		print("====== Check feature_vector(email) finish ======")

	def compute_feature_vector_all(self, data):
		"""updates the list of all feature vectors for given data"""
		filename = "output_data_"+ self.training_data_file + "_" + str(self.threshold) + ".csv"
		if path.isfile(filename):
			self.feature_vector_list = self.ctol(self.threshold, filename)
			return

		print("computing feature vectors")
		data.seek(0)
		self.feature_vector_list = []
		count = 0
		for email in data:
			count += 1
			if count % 200 == 0: print(count, "times looped")
			self.feature_vector_list.append(self.feature_vector(email))
		self.ltoc(self.feature_vector_list, self.threshold, filename)
		return

	def compute_label_all(self, data):
		"""updates list of labels for given data. assumes label is the first digit"""
		data.seek(0)
		print("saving labels")
		self.label_list = []
		for email in data:
			y = int(email[0]) * 2 - 1 # label of email, transform from [0,1] to [-1,1]
			self.label_list.append(y)
		if self.debug: print(self.label_list)

	def ltoc(self, list, threshold, filename):
		"""saves a csv from a list of list"""
		print("Saving list to", filename)
		csv = open(filename, "w")
		# csv.write(str(threshold) + '\n')
		for inner_list in list:
			string = ','.join(map(str, inner_list)) + '\n'
			csv.write(string)
		csv.close()
		return

	def ctol(self, threshold, filename):
		"""returns a list of list from csv"""
		print("Retrieving list from", filename)
		csv = open(filename, "r")
		listoflist = []
		for line in csv:
			inner_list = list(map(int, line.strip().split(',')))
			listoflist.append(inner_list)
		return listoflist

	def perceptron_train(self, data):
		"""trains perceptron and returns weight, mistakes, and iteration count. Assumes linear separability"""
		data.seek(0)
		self.compute_feature_vector_all(data)
		data.seek(0)
		self.compute_label_all(data)
		n = len(self.feature_vector_list)
		w = list(repeat(0, n)) # initialize  w as a zero vector
		k = 0 # number of mistakes
		iter = 0 # number of passes

		linearly_separated = False
		while (not linearly_separated) and iter < 30 :
			iter += 1
			linearly_separated = True
			for i in range(n):
				x = self.feature_vector_list[i]
				y = self.label_list[i]
				if y * dot_vec(w, x) > 0:
					pass # w = w
				else:
					k += 1
					linearly_separated = False
					w = add_vec(w, scal_vec(y, x))
			print(iter, "passes", k, "mistakes")
		print("completed training with", iter, "iterations and", k, "mistakes")
		return w, k, iter

	def perceptron_error(self, w, data):
		"""calculates the error rate, i.e., the fraction of examples that are misclassified by w"""

		pass



def main():
	# Create a perceptron of filename, threshold, and debug option
	p = Perceptron("train.txt", 21, False)
	data = open(p.training_data_file, "r")

	# 1a) open training data and load significant features into features
	# data.seek(0)
	p.features = p.words(data, p.threshold)
	if p.debug: p.check_words()

	# 1b) open training data and create feature vector for each and save them in a list
	data.seek(0)
	p.test_feature_vector = p.feature_vector(data.readline()) # one sample feature vector
	if p.debug: p.check_feature_vector()
	# data.seek(0)
	# p.compute_feature_vector_all(data)

	# 3a) Implement the function perceptron_train(data)
	# data.seek(0)
	w, k, iter = p.perceptron_train(data)
	# print(w)


	# 3b) Implement the function perceptron_error(w, data).

	# 4) test on validation.txt

	# 5) most positive/negative features
	significant = 14
	positive_weight = nlargest(significant, w)
	print(positive_weight)
	negative_weight = nsmallest(significant, w)
	# print(nsmallest(14, w))


if __name__ == "__main__":
	main()

