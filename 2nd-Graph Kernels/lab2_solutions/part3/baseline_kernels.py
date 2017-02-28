#!/usr/bin/env python


"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017

Graph Classification
"""

import networkx as nx
import numpy as np
import time
from collections import defaultdict
import copy


def compute_graphlet_kernel(graphs):
	"""
	Computes the graphlet kernel for connected graphlets of size 3
	and returns the kernel matrix.

	Parameters
	----------
	graphs : list
	A list of NetworkX graphs

	Returns
	-------
	K : numpy matrix
	The kernel matrix

	"""
	start_time = time.time()

	N = len(graphs)

	phi = np.zeros((N, 2))

	ind = 0
	for G in graphs:
		for node1 in G.nodes():
		    for node2 in G.neighbors(node1):
		        for node3 in G.neighbors(node2):
		            if node1 != node3:
		                if node3 in G.neighbors(node1):
		                	increment = 1.0/2.0
		                	phi[ind,0] += increment
		                else:
		                    increment = 1.0/6.0
		                    phi[ind,1] += increment

		ind += 1

	K = np.dot(phi,phi.T)
	end_time = time.time()
	print "Total time for Graphlet kernel: ", (end_time - start_time)

	return K



def compute_wl_subtree_kernel(graphs, h):
	"""
	Computes the Weisfeiler-Lehman kernel by performing h iterations
	and returns the kernel matrix.
	
	Parameters
  ----------
  graphs : list
    A list of NetworkX graphs
    
  h : int
	The number of WL iterations

  Returns
  -------
  K : numpy matrix
    The kernel matrix

  """
	for G in graphs:
		for node in G.nodes():
			G.node[node]['label'] = G.degree(node)
		
	start_time = time.time()

	labels = {}
	label_lookup = {}
	label_counter = 0

	N = len(graphs)

	orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

	# initial labeling
	ind = 0
	for G in graphs:
		labels[ind] = np.zeros(G.number_of_nodes(), dtype = np.int32)
		node2index = {}
		for node in G.nodes():
		    node2index[node] = len(node2index)
		    
		for node in G.nodes():
		    label = G.node[node]['label']
		    if not label_lookup.has_key(label):
		        label_lookup[label] = len(label_lookup)

		    labels[ind][node2index[node]] = label_lookup[label]
		    orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1
		
		ind += 1
		
	compressed_labels = copy.deepcopy(labels)

	# WL iterations
	for it in range(h):
		unique_labels_per_h = set()
		label_lookup = {}
		ind = 0
		for G in graphs:
		    node2index = {}
		    for node in G.nodes():
		        node2index[node] = len(node2index)
		        
		    for node in G.nodes():
		        node_label = tuple([labels[ind][node2index[node]]])
		        neighbors = G.neighbors(node)
		        if len(neighbors) > 0:
		            neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
		            node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
		        if not label_lookup.has_key(node_label):
		            label_lookup[node_label] = len(label_lookup)
		            
		        compressed_labels[ind][node2index[node]] = label_lookup[node_label]
		        orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1
		        
		    ind +=1
		    
		print "Number of compressed labels at iteration %s: %s"%(it, len(label_lookup))
		labels = copy.deepcopy(compressed_labels)

	K = np.zeros((N, N))

	for it in range(-1, h):
		for i in range(N):
		    for j in range(N):
		        common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][j].keys())
		        K[i][j] += sum([orig_graph_map[it][i].get(k,0)*orig_graph_map[it][j].get(k,0) for k in common_keys])

	end_time = time.time()
	print "Total time for WL subtree kernel: ", (end_time - start_time)
		                            
	return K
