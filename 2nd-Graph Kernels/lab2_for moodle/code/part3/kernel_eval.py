#!/usr/bin/env python


"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017

Graph Classification
"""

import numpy as np
import math
from sklearn import svm
from sklearn.model_selection import (KFold,ShuffleSplit,StratifiedShuffleSplit)
from sklearn.metrics import accuracy_score


def normalizekm(K):
	"""
	Normalizes the kernel matrix such that diagonal entries are equal to 1.
	
	Parameters
  ----------
  K : numpy matrix
    A kernel matrix

  Returns
  -------
  normalized_K : numpy matrix
    The normalized kernel matrix

  """
	v = np.sqrt(np.diag(K));
	nm =  np.outer(v,v)
	nm[np.where(nm==0)] = 1
	Knm = np.power(nm, -1)
	for i in range(K.shape[0]):
		for j in range(K.shape[1]):
			if np.isinf(Knm[i,j]):
				Knm[i,j] = 0
	normalized_K = K * Knm;
	return normalized_K

	
def svm_classification(K,labels):
	"""
	Given a set of kernel matrices, performs 10-fold cross-validation using an SVM and returns classification accuracy.
	At each iteration the optimal value of parameter C and the optimal kernel are determined using again cross-validation.
	
	Parameters
  ----------
  Ks : list
    A list of kernel matrices
    
  labels : list
    A list of class labels

  Returns
  -------
  result : dictionary
     A dictionary containing the accuracy, optimal value of C and optimal kernel for each iteration as well as the mean accuracy 
     and std of accuracies

  """
	# Number of instances
	n = len(labels)

	# Specify range of C values
	C_range = 10. ** np.arange(-5,6,2) / n

	# Number of folds
	cv = 10

	# Output variables
	result = {}
	result["opt_c"] = np.zeros(cv)
	result["accuracy"] = np.zeros(cv)
	
	labels = np.array(labels)

	kf = KFold(n_splits=10, shuffle=True, random_state=None)
	
	iteration = -1
	
	for train_indices_kf, test_indices_kf in kf.split(labels):
		iteration += 1
		
		num_c_vals = np.size(C_range)
		
		imresult = np.zeros(num_c_vals)
		
		labels_current = labels[train_indices_kf]
		K_normalized = normalizekm(K)
		K_current = K_normalized[np.ix_(train_indices_kf, train_indices_kf)]

		rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=None)
		for train_indices_ss, test_indices_ss in rs.split(train_indices_kf):
			K_train = K_current[np.ix_(train_indices_ss, train_indices_ss)]
			labels_train = labels_current[train_indices_ss]
			
			K_test = K_current[np.ix_(test_indices_ss, train_indices_ss)]
			labels_test = labels_current[test_indices_ss]
		
		for i in range(num_c_vals):
			# Train on 90% of 90%, predict on 10% of 90%
			clf = svm.SVC(C=C_range[i],kernel='precomputed')
			clf.fit(K_train, labels_train) 
			imresult[i] = clf.score(K_test, labels_test)
		 
		# Determine optimal C
		result["opt_c"][iteration] = C_range[np.argmax(imresult)]
				
		# Train on 90% with optimal kernel and C, predict on 10%
		K_normalized = normalizekm(K)
		K_train = K_normalized[np.ix_(train_indices_kf, train_indices_kf)]
		labels_train = labels[train_indices_kf]

		K_test = K_normalized[np.ix_(test_indices_kf, train_indices_kf)]
		labels_test = labels[test_indices_kf]
		
		clf = svm.SVC(C=result["opt_c"][iteration],kernel='precomputed')
		clf.fit(K_train, labels_train) 
		result["accuracy"][iteration] = clf.score(K_test, labels_test)
	
	result["mean_accuracy"] = np.mean(result["accuracy"]) 
	result["std"] = np.std(result["accuracy"])

	return result
