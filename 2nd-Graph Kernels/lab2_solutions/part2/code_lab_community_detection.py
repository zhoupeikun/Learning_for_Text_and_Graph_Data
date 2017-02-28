#!/usr/bin/env python

"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017

Community detection
"""

import os
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn import cluster

# you need to be in the directory where 'community_detection.py' is located
# to be able to import
# you can use os.chdir(path)
from community_detection import louvain

G = nx.read_edgelist("../dataset/CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())

# Get giant connected components (the biggest of the subgraphs)
GCC = max(nx.connected_component_subgraphs(G), key=len)

# Spectral clustering algorithm
# Implement and apply spectral clustering

def spectral_clustering(G, k):
    #L = nx.laplacian_matrix(G).astype(float) # Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float) # Normalized Laplacian
    eigval, eigvec = eigs(L,k=k, which="SR") # Calculate eigenvalues and eigenvectors
    eigval = eigval.real # Keep the real part
    eigvec = eigvec.real # Keep the real part
    # sort is implemented by default in increasing order
    idx = eigval.argsort() # Get indices of sorted eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    km = cluster.KMeans(n_clusters=k, init='random').fit(eigvec)
    membership = list(km.labels_)
    # will contain node IDs as keys and membership as values
    clustering = {}
    nodes = G.nodes()
    for i in range(len(nodes)):
        clustering[nodes[i]] = membership[i]
    
    return clustering
	
# Apply spectral clustering to the CA-HepTh dataset
clustering = spectral_clustering(G=GCC, k=60)

# sanity check
GCC.number_of_nodes() == len(clustering)

# Modularity
# Implement and compute it for two clustering results

# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    n_clusters = len(list(set(clustering.values())))
    modularity = 0 # Initialize total modularity value
    #Iterate over all clusters
    for i in range(n_clusters):
        # Get the nodes that belong to the i-th cluster
        nodeList = [n for n,v in clustering.iteritems() if v == i]
        # Create the subgraphs that correspond to each cluster				
        subG = G.subgraph(nodeList)
        temp1 = nx.number_of_edges(subG) / float(nx.number_of_edges(G))
        temp2 = pow(sum(nx.degree(G, nodeList).values()) / float(2 * nx.number_of_edges(G)), 2)
        modularity = modularity + (temp1 - temp2)
    return modularity
	
print "Modularity Spectral Clustering: ", modularity(GCC, clustering)

# Implement random clustering
k = 60
r_clustering = {}
for node in GCC.nodes():
    r_clustering[node] = randint(0,k-1)
	
print "Modularity Random Clustering: ", modularity(GCC, r_clustering)

# Louvain
# Run it and compute modularity

# Partition graph using the Louvain method
clustering = louvain(GCC)

print "Modularity Louvain: ", modularity(GCC, clustering)
