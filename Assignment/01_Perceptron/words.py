# Simon Seo, 2016.Feb.06, Perceptron Assignment
# This is an implementation of the Perceptron algorithm for classifying spam emails

### Global Variables ###
features = [] # vector of features in consideration

### End of Global Variables ###

# words(data, X)
# takes train.txt as input and outputs a Python list
# containing all the words that occur in at least 20 emails.
# assumes that each email is a line of string where the first column is the label 0 or 1.
#
# @data: spam training data file
# @X: overfitting threshold
def words(data, X):
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

# Assumes that features is defined properly
# 
# @email: string that includes the label 
def feature_vector(email):
    if not len(features) > 0: # if features is not defined
        raise MyLookupException({"message":"variable has not been initiated", "variable":"features"})
    email = email.split()
    email.pop(0)  # remove label
    feature_vector = []
    for word in features:
        if word in email:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    return feature_vector


class MyLookupException(Exception):
    """Raise for uninitiated variables exception"""

# main

# 1a) open training data and load significant features into features
data = open("train.txt", "r")
features = words(data, 20)
data.close()

# check
print("====== Check words(data, X) start ======")
print(features)
print('length', len(features))
print('httpaddr', 'httpaddr' in features)
print('1', '1' in features)
print('0', '0' in features)
print("====== Check words(data, X) finish ======")

# 1b) open training data and create feature vector for each and save them in a list
data = open("train.txt", "r")
test_feature_vector = feature_vector(data.readline()) # one sample feature vector
'''
feature_vector_list = []
for email in data:
    feature_vector_list.append(feature_vector(email))
'''
data.close()

# check
print("====== Check feature_vector(email) start ======")
print("feature vector for one email: ")
print(test_feature_vector)
data = open("train.txt", "r")
email = data.readline().split()
data.close()
for i in range(len(test_feature_vector)):
    el = test_feature_vector[i]
    if el == 1:
        if not (features[i] in email):
            print("feature vector purity test failed")
            break
        else:
            print(features[i])
print("feature vector purity test passed")
print("====== Check feature_vector(email) finish ======")