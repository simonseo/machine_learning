import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Read the data ##########

faces = np.genfromtxt('faces.csv', dtype=int, delimiter=',')
temp = np.empty([400, 64, 64])
for i in range(400):
	for j in range(64):
		for k in range(64):
			temp[i][k][j] = faces[i][64*j+k]
faces = np.reshape(temp, (400, 64, 64))


######### Global Variable ##########

FACE_COUNT = faces.shape[0]
SHOW_PREVIOUS_WORK = False
image_count = 0

######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.

def normalize(U):
	return U / LA.norm(U) 

######### Function that wraps matplotlib.pyplot.imshow #########

def spawn_imshow_wrapper():
	image_count = 0
	def show(image, title=None):
		nonlocal image_count
		image_count += 1
		plt.figure(image_count)
		plt.title('Image #{}'.format(image_count) if title is None else title)
		plt.imshow(image, cmap=plt.cm.gray)
	return show
show = spawn_imshow_wrapper()


######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.

if SHOW_PREVIOUS_WORK:
	first_face = faces[0]
	show(first_face, title='First_face')


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.

#### Your Code Here ####

if SHOW_PREVIOUS_WORK:
	random_mask = np.random.choice(FACE_COUNT, 5)
	random_faces = faces[random_mask]
	for face in random_faces:
		show(face)



########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over 
#   the flattened array by default, otherwise over the specified axis. 
#   float64 intermediate and return values are used for integer inputs.

#### Your Code Here ####

mean_face = np.mean(faces, axis=0)
if SHOW_PREVIOUS_WORK:
	show(mean_face, title='Mean Face of all 400 faces')

######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####

centered_faces = faces - np.repeat([mean_face], FACE_COUNT, axis=0)
if SHOW_PREVIOUS_WORK:
	show(centered_faces[0])

######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####




########## Display the first 10 principal components ##################

#### Your Code Here ####




########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####





########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####




######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####




plt.show()
