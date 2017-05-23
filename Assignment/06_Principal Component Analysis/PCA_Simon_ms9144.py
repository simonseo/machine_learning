#!/usr/bin/python
# -*- coding: utf-8 -*- 
# @File Name: PCA_Simon_ms9144.py
# @Created:   2017-05-12 15:44:15  Simon Myunggun Seo (simon.seo@nyu.edu) 
# @Updated:   2017-05-15 11:12:46  Simon Seo (simon.seo@nyu.edu)

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

######### Global Variables #########

FACE_COUNT = 400
SHOW_PREVIOUS_WORK = False



######### Read the data and change it to a row-first order (from column first) ##########

faces = np.genfromtxt('faces.csv', dtype=int, delimiter=',')
faces = np.reshape(faces.T, (64,64,FACE_COUNT), order='F')
faces = np.reshape(faces, (4096, FACE_COUNT))
faces = np.reshape(faces.T, (FACE_COUNT, 64, 64))



######### Function that normalizes a vector x (i.e. |x|=1 ) #########
def normalize(U):
	return U / LA.norm(U)

######### Function that wraps matplotlib.pyplot.imshow #########
def spawn_imshow_wrapper():
	image_count = 1
	def show_q(image, title=None, subplot=(0,0,0)):
		nonlocal image_count
		if subplot == (0,0,0):
			plt.figure(image_count)
		else:
			if subplot[2] == 1:
				plt.figure(image_count)
			plt.subplot(*subplot)
		plt.axis('off')
		plt.title('Image #{}'.format(image_count) if title is None else title)
		plt.imshow(image, cmap=plt.cm.gray)
	def show():
		nonlocal image_count
		image_count += 1
		plt.show()
	return show_q, show
show_q, show = spawn_imshow_wrapper()

######### Functions for formating np arrays #########
def square(arr):
	return np.reshape(np.array(arr), (64,64))
def flat(arr):
	return np.reshape(np.array(arr), (-1))




######### Display first face #########

if SHOW_PREVIOUS_WORK:
	first_face = faces[0]
	show_q(first_face, title='First_face')
	show()



########## display a random face ###########

if SHOW_PREVIOUS_WORK:
	random_mask = np.random.choice(FACE_COUNT, 1)
	random_faces = faces[random_mask]
	for face in random_faces:
		show_q(face, title='Random face')
	show()



########## compute and display the mean face ###########

mean_face = np.mean(faces, axis=0)
if SHOW_PREVIOUS_WORK:
	show_q(mean_face, title='Mean Face of all {} faces'.format(FACE_COUNT))
	show()



######### substract the mean from the face images and get the centralized data matrix A ###########

centered_faces = faces - np.repeat([mean_face], FACE_COUNT, axis=0)
if SHOW_PREVIOUS_WORK:
	show_q(centered_faces[0], title='Centered Face of first face')
	show()



######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

A = flat_faces = np.reshape(centered_faces, (FACE_COUNT,-1)) # flattened each face into a row vector (400,4096)
L = np.matmul(A, A.T) #(400,400)
eigvals, v = LA.eig(L) # v = (400,400)
z = np.matmul(A.T, v) # (4096,400)
z = z.T
eig = [{'vec': z[i], 'val': eigvals[i] } for i in range(FACE_COUNT)]
eig.sort(reverse=True, key=lambda eig: eig['val'])
z = np.array([normalize(eig[i]['vec']) for i in range(FACE_COUNT)])



########## Display the first 10 principal components ##################

if SHOW_PREVIOUS_WORK:
	k = 10
	for i in range(k):
		pc_face = z[i]
		pc_face = square(pc_face)
		show_q(pc_face, subplot=((k+4)//5, 5, i+1), title='PC #{}'.format(i+1))
	plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.5)
	show()



########## Reconstruct the first face using the first two PCs #########

if SHOW_PREVIOUS_WORK:
	first_face = flat(centered_faces[0]) # (4096)
	show_q(square(faces[0]), subplot=(1,2,1), title='first face')
	U = z[:2] #(2, 4096)
	weights = np.matmul(U, first_face.T) # (2, 1)
	reconst_face = flat(mean_face) + np.matmul(U.T, weights) 

	show_q(square(reconst_face), subplot=(1,2,2), title='reshaped')
	show()



########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

if SHOW_PREVIOUS_WORK:
	k_list = [5, 10, 25, 50, 100, 200, 300, 399]
	l = len(k_list)
	random_mask = np.random.choice(FACE_COUNT, 1)
	random_faces = flat_faces[random_mask]
	for face in random_faces:
		reconst_face = flat(mean_face)
		for i in range(k_list[-1]): 
			# Matrix multiplication step by step to save computation in case k_list becomes larger
			pc_face = flat(z[i])
			reconst_face += np.dot(face, pc_face) * pc_face
			if i+1 in k_list:
				show_q(square(reconst_face), subplot=((l+4)//5, 5, k_list.index(i+1)+1), title='{} PCs'.format(i+1))
		show_q(square(face) + mean_face, subplot=((l+4)//5, 5, l+1), title='Original face')
		plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.5)
		show()



######### Plot proportion of variance of all the PCs ###############

if SHOW_PREVIOUS_WORK or True:
	# data
	eigvals = np.array([e['val'] for e in eig[:-1]]) # eig was sorted above
	print(eigvals[FACE_COUNT-5:], sum(eigvals))
	x = range(FACE_COUNT - 1)
	y = eigvals/sum(eigvals)
	y_comp = [-(0.25/400)*i + 0.25 for i in range(0,FACE_COUNT - 1)]

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
	plt.yscale('log')
	plt.title('Proportion of variance of all {} PCs (log)'.format(FACE_COUNT))
	plt.grid(True)

	plt.subplots_adjust(hspace=0.5, wspace=0.35)
	show()

