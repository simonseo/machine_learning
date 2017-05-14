import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


######### Global Variable ##########

FACE_COUNT = 400
SHOW_PREVIOUS_WORK = False
image_count = 0

######### Read the data and change it to a row-first order (from column first) ##########

faces = np.genfromtxt('faces.csv', dtype=int, delimiter=',')
faces = np.reshape(faces.T, (64,64,FACE_COUNT), order='F')
faces = np.reshape(faces, (4096, FACE_COUNT))
faces = np.reshape(faces.T, (FACE_COUNT, 64, 64))


######### Function that normalizes a vector x (i.e. |x|=1 ) #########

# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False) 
#   This function is able to return one of eight different matrix norms, 
#   or one of an infinite number of vector norms (described below), 
#   depending on the value of the ord parameter.
#   
#   row by row normalization

def normalize(U):
	return U / LA.norm(U)

######### Function that wraps matplotlib.pyplot.imshow #########

def spawn_imshow_wrapper():
	image_count = 0
	def show_q(image, title=None):
		nonlocal image_count
		image_count += 1
		plt.figure(image_count)
		plt.title('Image #{}'.format(image_count) if title is None else title)
		plt.imshow(image, cmap=plt.cm.gray)
	def show():
		plt.show()
	return show_q, show
show_q, show = spawn_imshow_wrapper()

######### Formats into 64x64 square image #########
def square(arr):
	return np.reshape(np.array(arr), (64,64))
def flat(arr):
	return np.reshape(np.array(arr), (-1))

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
	show_q(first_face, title='First_face')
	show()


########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.

#### Your Code Here ####

if SHOW_PREVIOUS_WORK:
	random_mask = np.random.choice(FACE_COUNT, 1)
	random_faces = faces[random_mask]
	for face in random_faces:
		show_q(face, title='Random face')
	show()



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
	show_q(mean_face, title='Mean Face of all {} faces'.format(FACE_COUNT))
	show()

######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

#### Your Code Here ####

centered_faces = faces - np.repeat([mean_face], FACE_COUNT, axis=0)
if SHOW_PREVIOUS_WORK:
	show_q(centered_faces[0], title='Centered Face')
	show()

######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data. 
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations. 
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).
#   
# > numpy.cov(m, y=None, rowvar=True)

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity. 
#   The eigenvalues are not necessarily ordered. 

#### Your Code Here ####


A = flat_faces = np.reshape(centered_faces, (FACE_COUNT,-1)) # flattened each face into a row vector (400,4096)
L = np.matmul(A, A.T) #(400,400)
eigvals, v = LA.eig(L) # v = (400,400)
z = np.matmul(A.T, v) # (4096,400)
z = z.T
eig = [{'vec': z[i], 'val': eigvals[i] } for i in range(FACE_COUNT)]
eig.sort(reverse=True, key=lambda eig: eig['val'])
z = np.array([normalize(eig[i]['vec']) for i in range(FACE_COUNT)])







########## Display the first 10 principal components ##################

#### Your Code Here ####
if SHOW_PREVIOUS_WORK:
	for i in range(10):
		pc_face = z[i]
		pc_face = square(pc_face)
		show_q(pc_face, title='Principal Component #{}'.format(i+1))
	show()


########## Reconstruct the first face using the first two PCs #########

#### Your Code Here ####
if SHOW_PREVIOUS_WORK:
	first_face = flat(centered_faces[0]) # (4096)
	show_q(square(faces[0]), title='first face')
	U = z[:2] #(2, 4096)
	weights = np.matmul(U, first_face.T) # (2, 1)
	reconst_face = flat(mean_face) + np.matmul(U.T, weights) 

	show_q(square(reconst_face), title='first face reshaped')
	show()



########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####
if SHOW_PREVIOUS_WORK:
	k_list = [5, 10, 25, 50, 100, 200, 300, 399]
	random_mask = np.random.choice(FACE_COUNT, 1)
	random_faces = flat_faces[random_mask]
	for face in random_faces:
		reconst_face = flat(mean_face)
		for i in range(k_list[-1]): 
			# Matrix multiplication step by step to save computation in case k_list becomes larger
			pc_face = flat(z[i])
			reconst_face += np.dot(face, pc_face) * pc_face
			if i+1 in k_list:
				show_q(square(reconst_face), title='Reconstruction using {} PCs'.format(i+1))
		show_q(square(face) + mean_face, title='Original face')
		show()


######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes. 
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure. 
#   When running in ipython with its pylab mode, 
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
# data
eigvals = np.array([eig[i]['val'] for i in range(FACE_COUNT)]) # eig was sorted above
x = range(FACE_COUNT)
y = eigvals/sum(eigvals)
y_comp = [-(0.25/400)*i + 0.25 for i in range(1,401)]

# linear
plt.subplot(211)
plt.plot(x, y, label='Variance proportion')
plt.plot(x, y_comp, label=r'Reference: $0.25x + 400y = 100$')
plt.legend()
plt.yscale('linear')
plt.title('Proportion of variance of all {} PCs (linear)'.format(FACE_COUNT))
plt.grid(True)


# log
plt.subplot(212)
plt.plot(x, y, label='Variance proportion')
plt.plot(x, y_comp, label=r'Reference: $0.25x + 400y = 100$')
plt.legend()
plt.yscale('log')
plt.title('Proportion of variance of all {} PCs (log)'.format(FACE_COUNT))
plt.grid(True)

plt.subplots_adjust(hspace=0.5, wspace=0.35)
show()

