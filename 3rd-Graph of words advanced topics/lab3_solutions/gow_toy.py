import string
import re 
import itertools
import copy
import igraph
import nltk
import operator
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag

# might also be required:
# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('stopwords')

# import custom functions
from library import clean_text_simple, terms_to_graph, unweighted_k_core

my_doc = '''A method for solution of systems of linear algebraic equations 
with m-dimensional lambda matrices. A system of linear algebraic 
equations with m-dimensional lambda matrices is considered. 
The proposed method of searching for the solution of this system 
lies in reducing it to a numerical system of a special kind.'''

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc)
                              
g = terms_to_graph(my_tokens, w=4)
    
# number of edges
len(g.es)

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

edge_weights

for w in range(2,11):
    g = terms_to_graph(my_tokens, w)
    print g.density()
    
# decompose g
g = terms_to_graph(my_tokens, w=4)
core_numbers = unweighted_k_core(g)

# compare with igraph method
dict(zip(g.vs["name"],g.coreness()))

# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]