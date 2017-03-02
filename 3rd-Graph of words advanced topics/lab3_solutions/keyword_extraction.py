import os
import random
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
from sklearn.feature_extraction.text import TfidfVectorizer

# might also be required:
# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('stopwords')

from library import clean_text_simple, terms_to_graph, accuracy_metrics

stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('english')

##################################
# read and pre-process abstracts #
##################################

# to fill
# path_to_abstracts = 
abstract_names = sorted(os.listdir(path_to_abstracts))

abstracts = []
counter = 0

for filename in abstract_names:
    # read file
    with open(path_to_abstracts + '\\' + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text =  re.sub('\s+', ' ', text)
    abstracts.append(text)
    
    counter += 1
    if counter % 100 == 0:
        print counter, 'files processed'
     

abstracts_cleaned = []
counter = 0

for abstract in abstracts:
    my_tokens = clean_text_simple(abstract)
    abstracts_cleaned.append(my_tokens)
    counter += 1
    if counter % 100 == 0:
        print counter, 'abstracts processed'
                       
#################################
# read and pre-process keywords #
#################################

# to fill
# path_to_keywords = 
keywords_names = sorted(os.listdir(path_to_keywords))
   
keywords_gold_standard = []
counter = 0

for filename in keywords_names:
    # read file
    with open(path_to_keywords + '\\' + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text =  re.sub('\s+', ' ', text)
    # convert to lower case
    text = text.lower()
    # turn string into list of keywords, preserving intra-word dashes 
    # but breaking n-grams into unigrams to easily compute precision and recall
    keywords = text.split(';')
    keywords = [keyword.strip().split(' ') for keyword in keywords]
    # flatten list
    keywords = [keyword for sublist in keywords for keyword in sublist]
    # remove stopwords (rare but can happen due to n-gram breaking)
    keywords = [keyword for keyword in keywords if keyword not in stpwds]
    # apply Porter's stemmer
    keywords_stemmed = [stemmer.stem(keyword) for keyword in keywords]
    # remove duplicates (can happen due to n-gram breaking)
    keywords_stemmed_unique = list(set(keywords_stemmed))
    
    keywords_gold_standard.append(keywords_stemmed_unique)
    
    counter += 1
    if counter % 100 == 0:
        print counter, 'files processed'
    
###############################
# keyword extraction with gow #
###############################
   
keywords_gow = []
counter = 0   
   
for abstract in abstracts_cleaned:
    # create graph-of-words
    g = terms_to_graph(abstract, w=4)
    # decompose graph-of-words
    core_numbers = dict(zip(g.vs['name'],g.coreness()))
    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd, c_n in core_numbers.iteritems() if c_n == max_c_n]
    # save results
    keywords_gow.append(keywords)
    
    counter += 1
    if counter % 100 == 0:
        print counter, 'abstracts processed'

#########################################
# keyword extraction with the baselines #
#########################################

# TF_IDF

# to ensure same pre-processing as the other methods
abstracts_cleaned_strings = [' '.join(elt) for elt in abstracts_cleaned]

tfidf_vectorizer = TfidfVectorizer(stop_words = stpwds)
doc_term_matrix = tfidf_vectorizer.fit_transform(abstracts_cleaned_strings)
terms = tfidf_vectorizer.get_feature_names()
vectors_list = doc_term_matrix.todense().tolist()

keywords_tfidf = []
counter = 0

for vector in vectors_list:
    
    # bow feature vector as list of tuples
    terms_weights = zip(terms,vector)
    # keep only non zero values (the words in the document)
    nonzero = [tuple for tuple in terms_weights if tuple[1]!=0]
    # rank by decreasing weights
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True)
    # retain top 33% words as keywords
    numb_to_retain = int(round(len(nonzero)/3))
    keywords = [tuple[0] for tuple in nonzero[:numb_to_retain]]
    
    keywords_tfidf.append(keywords)
    
    counter += 1
    if counter % 100 == 0:
        print counter, 'vectors processed'

# PageRank

keywords_pr = []
counter = 0

for abstract in abstracts_cleaned:
    # create graph-of-words
    g = terms_to_graph(abstract, w=4)
    # compute PageRank scores
    pr_scores = zip(g.vs['name'],g.pagerank())
    # rank in decreasing order
    pr_scores = sorted(pr_scores, key=operator.itemgetter(1), reverse=True)
    # retain top 33% words as keywords
    numb_to_retain = int(round(len(pr_scores)/3))
    keywords = [tuple[0] for tuple in pr_scores[:numb_to_retain]]
    
    keywords_pr.append(keywords)    
    
    counter += 1
    if counter % 100 == 0:
        print counter, 'abstracts processed'

# Random

keywords_rand = []
counter = 0

for abstract in abstracts_cleaned:
    # retain 33% of the words as keywords (randomly selected without replacement)
    numb_to_retain = int(round(len(abstract)/3))
    ind_to_retain = random.sample(range(len(abstract)), numb_to_retain)    
    keywords = [abstract[ind] for ind in ind_to_retain]
    
    keywords_rand.append(keywords)    
    
    counter += 1
    if counter % 100 == 0:
        print counter, 'abstracts processed'

##########################
# performance evaluation #
##########################

perf_gow= []
perf_tfidf = []
perf_pr = []
perf_rand = []

for ind, truth in enumerate(keywords_gold_standard):
    perf_gow.append(accuracy_metrics(keywords_gow[ind], truth))
    perf_tfidf.append(accuracy_metrics(keywords_tfidf[ind], truth))
    perf_pr.append(accuracy_metrics(keywords_pr[ind], truth))
    perf_rand.append(accuracy_metrics(keywords_rand[ind], truth))
    
lkgs = len(keywords_gold_standard)

# macro-averaged results (averaged at the collection level)

results = {'gow':perf_gow,'tfidf':perf_tfidf,'pr':perf_pr, 'rand':perf_rand}

for name, result in results.iteritems():
    print name + ' performance: \n'
    print 'precision:', sum([tuple[0] for tuple in result])/lkgs
    print 'recall:', sum([tuple[1] for tuple in result])/lkgs
    print 'F-1 score:', sum([tuple[2] for tuple in result])/lkgs
    print '\n'