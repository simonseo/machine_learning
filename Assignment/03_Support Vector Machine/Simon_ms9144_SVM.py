from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from numpy import sum

def main():
	# preprocess training data
	train_filename = 'mnist_train.txt'
	X_train, Y_train = file_to_vector(train_filename)
	X_train = normalize(X_train)


	# train classifier
	classifier = SVC(C=1.5, kernel='rbf', gamma=0.005) # optimal C=1.5, gamma=0.005
	classifier.fit(X_train, Y_train)

	# preprocess test data
	test_filename = 'mnist_test.txt'
	X_test, Y_test = file_to_vector(test_filename)
	X_test = normalize(X_test)


	# make predictions on test set
	prediction = classifier.predict(X_test)

	# report scores
	l = len(prediction)
	score = 100 - sum([int(prediction[i]==Y_test[i]) for i in range(l)]) * 100 / l
	print('test data score (%)', score)
	
	score = sum([(1 - score)*100 for score in cross_val_score(classifier, X_train, Y_train, cv=5)])/5
	print('cross validation score (%)', score)


def normalize(V_2D):
	'''maps a 2D array of values 0~255 to that of -1.0~1.0'''
	return [[int(x) * 2 / 255 - 1 for x in V] for V in V_2D]
	
def file_to_vector(filename):
	'''opens a file and return the feature vectors and labels'''
	file = open(filename, 'r')
	X = []
	Y = []
	for line in file:
		line = line.strip().split(',')
		Y.append(int(line.pop(0)))
		X.append(line)
	file.close()
	return X, Y

if __name__ == '__main__':
	main()