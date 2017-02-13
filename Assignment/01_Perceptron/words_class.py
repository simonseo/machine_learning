class Perceptron():
	"""implementation of the Perceptron algorithm for classifying spam emails"""
	def __init__(self, training_data):
		self.training_data = training_data
		self.features = []

		# 1a) open training data and load significant features into features
		data = open(self.training_data, "r")
		self.features = self.words(data, 20)
		data.close()
		self.check_words()

		# 1b) open training data and create feature vector for each and save them in a list
		data = open(self.training_data, "r")
		self.test_feature_vector = self.feature_vector(data.readline()) # one sample feature vector
		# self.feature_vector_list = []
		# for email in data:
		#     self.feature_vector_list.append(self.feature_vector(email))
		data.close()
		self.check_feature_vector()


	# words(data, X)
	# takes train.txt as input and outputs a Python list
	# containing all the words that occur in at least X emails.
	# assumes that each email is a line of string where the first column is the label 0 or 1.
	#
	# @data: spam training data file
	# @X: overfitting threshold
	def words(self, data, X):
	    word_occurrence = {}

	    for email in data:
	        email = email.split()
	        email.pop(0)  # remove label
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
		print("====== Check words(data, X) start ======")
		print(self.features)
		print('length', len(self.features))
		print('httpaddr', 'httpaddr' in self.features)
		print('1', '1' in self.features)
		print('0', '0' in self.features)
		print("====== Check words(data, X) finish ======")


	# Assumes that features is defined properly
	# 
	# @email: string that includes the label 
	def feature_vector(self, email):
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
		print("====== Check feature_vector(email) start ======")
		print("feature vector for one email: ")
		print(self.test_feature_vector)
		data = open(self.training_data, "r")
		email = data.readline().split()
		data.close()
		for i in range(len(self.test_feature_vector)):
		    el = self.test_feature_vector[i]
		    if el == 1:
		        if not (self.features[i] in email):
		            print("feature vector purity test failed")
		            break
		        else:
		            print(self.features[i])
		print("feature vector purity test passed")
		print("====== Check feature_vector(email) finish ======")

# Create a perceptron
p = Perceptron("train.txt")
