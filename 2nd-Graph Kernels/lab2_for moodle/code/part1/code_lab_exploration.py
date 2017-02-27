#!/usr/bin/env python


"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017
"""

# Import modules
from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.sparse as sparse
from numpy import *
import numpy.linalg


############## Question 1
# Load the graph into an undirected NetworkX graph

##################
# your code here #
##################

# hint: use read_edgelist function of NetworkX

G=nx.read_edgelist("../dataset/CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())


############## Question 2
# Network Characteristics
print 'Number of nodes:', G.number_of_nodes() 
print 'Number of edges:', G.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(G)




# Get giant connected component (GCC)

##################
# your code here #
##################

# hint: use connected_component_subgraphs function of NetworkX, GCC is the biggest of the subgraphs
GCC=list(nx.connected_component_subgraphs(G))[0]


# Compute the fraction of nodes and edges in GCC 

##################
# your code here #
##################
print 'Fraction of nodes in GCC', GCC.number_of_nodes() / G.number_of_nodes()
print 'Fraction of edges in GCC', GCC.number_of_edges() / G.number_of_edges()



############## Question 3
# Extract degree sequence and compute min, max, median and mean degree

##################
# your code here #
##################

# hint: use the min, max, median and mean functions of NumPy
# degree
degree_sequence = G.degree().values()
print 'Min degree', np.min(degree_sequence)
print 'Max degree', np.max(degree_sequence)
print 'Median degree', np.median(degree_sequence)
print 'Mean degree', np.mean(degree_sequence)

# Degree distribution
y=nx.degree_histogram(G)
plt.figure(1)
plt.plot(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
f.savefig("degree.png",format="png")

plt.figure(2)
plt.loglog(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
s.savefig("degree_loglog.png",format="png")
