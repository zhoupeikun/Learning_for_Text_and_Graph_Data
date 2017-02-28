#!/usr/bin/env python


"""
Graph Mining and Analysis with Python - Master Data Science - MVA - Feb 2017
"""

#%%
# Import modules
from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.sparse as sparse
from numpy import *
import numpy.linalg

G=nx.read_edgelist("../dataset/CA-HepTh.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())


############## Question 2
# Network Characteristics
print 'Number of nodes:', G.number_of_nodes() 
print 'Number of edges:', G.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(G)

# Connected components
GCC=list(nx.connected_component_subgraphs(G))[0]

# Fraction of nodes and edges in GCC 
print "Fraction of nodes in GCC: ", GCC.number_of_nodes() / G.number_of_nodes()
print "Fraction of edges in GCC: ", GCC.number_of_edges() / G.number_of_edges()

#%%
############## Question 3
# Degree
degree_sequence = G.degree().values()
print "Min degree ", np.min(degree_sequence)
print "Max degree ", np.max(degree_sequence)
print "Median degree ", np.median(degree_sequence)
print "Mean degree ", np.mean(degree_sequence)

# Degree distribution
y=nx.degree_histogram(G)
plt.figure(1)
plt.plot(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#f.savefig("degree.png",format="png")

plt.figure(2)
plt.loglog(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#s.savefig("degree_loglog.png",format="png")
