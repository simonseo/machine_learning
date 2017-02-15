from os import path
from heapq import nlargest, nsmallest
import numpy as np
from re import search

'''To do list
implement CLI

'''


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
	def __init__(self, training_data_origin, training_size, threshold, iter_limit, debug):
		print("Perceptrons! Roll out!")
		self.training_data_origin = training_data_origin
		self.training_size = training_size
		self.threshold = threshold
		self.iter_limit = iter_limit
		self.debug = debug

		self.filename_train = ''
		self.filename_validation = ''
		self.filename_test = ''

		self.features = [] # list of features in consideration
		self.feature_vector_list = [] # list of feature vectors of each email
		self.label_list = [] # list of labels for each email

	def preprocess(self):
		"""cuts training data into training_size training data and the rest into a validation set"""
		filename_train = "train_N" + str(self.training_size) + ".txt"
		filename_validation = "validation_N" + str(self.training_size) + ".txt"

		if path.isfile(filename_train) and path.isfile(filename_validation):
			print("training and validation set found for N =", self.training_size)
			self.filename_train = filename_train
			self.filename_validation = filename_validation
			return

		def find_nth(haystack, needle, n):
			start = haystack.find(needle)
			while start >= 0 and n > 1:
				start = haystack.find(needle, start+len(needle))
				n -= 1
			return start

		data = open(self.training_data_origin, "r")
		text = data.read()
		index = find_nth(text, '\n', self.training_size)

		train = open(filename_train, "w")
		train.write(text[:index+1])
		train.close()

		validation = open(filename_validation, "w")
		validation.write(text[index+1:])
		validation.close()

		data.close()

		self.filename_train = filename_train
		self.filename_validation = filename_validation

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
		data = open(self.filename_train, "r")
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
		filename = search(r"name='(.+)' mode", str(data)).group(1)
		filename += "_"+ str(self.training_size) + "_" + str(self.threshold) + ".csv"
		if path.isfile(filename):
			return self.ctol(self.threshold, filename)

		print("computing feature vectors for", filename)
		data.seek(0)
		# self.feature_vector_list = []
		feature_vector_list = []
		count = 0
		for email in data:
			count += 1
			if count % 200 == 0: print(count, "times looped")
			feature_vector_list.append(self.feature_vector(email))
		self.ltoc(feature_vector_list, self.threshold, filename)
		return feature_vector_list

	def compute_label_all(self, data):
		"""updates list of labels for given data. assumes label is the first digit"""
		data.seek(0)
		print("saving labels")
		label_list = []
		for email in data:
			y = int(email[0]) * 2 - 1 # label of email, transform from [0,1] to [-1,1]
			label_list.append(y)
		if self.debug: print(label_list)
		return label_list

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
		csv.close()
		return listoflist

	def perceptron_train(self, data):
		"""trains perceptron and returns weight, mistakes, and iteration count. Assumes linear separability"""
		data.seek(0)
		self.feature_vector_list = self.compute_feature_vector_all(data)
		data.seek(0)
		self.label_list = self.compute_label_all(data)
		n = len(self.feature_vector_list[0])
		w = np.zeros(n) # initialize  w as a zero vector
		k = 0 # number of mistakes
		iter = 0 # number of passes

		linearly_separated = False
		while (not linearly_separated) and iter < self.iter_limit :
			iter += 1
			linearly_separated = True
			for i in range(n):
				if iter == 1:
					x = np.array(self.feature_vector_list[i])
					self.feature_vector_list[i] = x
				else:
					x = self.feature_vector_list[i]
				y = self.label_list[i]
				if y * np.dot(w, x) > 0:
					pass # w = w
				else:
					k += 1
					linearly_separated = False
					w = w + y * x
			print(iter, "passes", k, "mistakes")
		print("completed training with", iter, "iterations and", k, "mistakes")
		return w, k, iter

	def perceptron_error(self, w, test_set):
		"""calculates the error rate, i.e., the fraction of examples that are misclassified by w"""
		#test_set called using p.filename_validation or a test file.
		data = open(test_set, "r")
		data.seek(0)
		feature_vector_list = self.compute_feature_vector_all(data)
		data.seek(0)
		label_list = self.compute_label_all(data)
		n = len(feature_vector_list[0])
		k = 0

			for i in range(n):
				x = np.array(feature_vector_list[i])
				y = label_list[i]
				if y * np.dot(w, x) > 0:
					pass # w = w
				else:
					k += 1
		print(k, "misclassified out of", n, "emails. Error rate:", 100*k/n, "%")
		data.close()
		return k/n



def main():
	# Create a perceptron of filename, N, threshold, iteration limit and debug option
	p = Perceptron("spam_train.txt", 4000, 18, 40, False)
	p.preprocess()

	# 1a) open training data and load significant features into features
	data = open(p.filename_train, "r")
	p.features = p.words(data, p.threshold)
	if p.debug: p.check_words()

	# 1b) open training data and create feature vector for each and save them in a list
	# p.compute_feature_vector_all(data) # this line is commented out because it is done in 2a
	if p.debug: 
		data.seek(0)
		p.test_feature_vector = p.feature_vector(data.readline()) # one sample feature vector
		p.check_feature_vector()

	# 2a) Implement the function perceptron_train(data)
	w, k, iter = p.perceptron_train(data)
	if p.debug: print(w)

	# 2b) Implement the function perceptron_error(w, data).
	print("train", p.perceptron_error(p.filename_train))
	print("validation", p.perceptron_error(p.filename_validation))

	# 3) Validate on both training data and on validation data

	# 4) 12 most positive/negative features
	significant = 12
	positive_weight = nlargest(significant, w)
	print(positive_weight)
	negative_weight = nsmallest(significant, w)
	# print(nsmallest(14, w))

	# 5) Plot validation error as a function of N (N = 100, 200, 400, 800, 2000, 4000)

	# 6) Plot iter as a function of N (N = 100, 200, 400, 800, 2000, 4000)
	
	# 7) add an argument to limit the max number of passes

	# 8) validation error for different configurations

	# 9) If X = 1200, is it linearly separable? how many features?

	# 10) why separate training, validation, and test
	data.close()

if __name__ == "__main__":
	main()

