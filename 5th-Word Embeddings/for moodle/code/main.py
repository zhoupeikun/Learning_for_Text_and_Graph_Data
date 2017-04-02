import re
import string
import shelve
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
from gensim.models.word2vec import Word2Vec

# remove dashes and apostrophes from punctuation marks 
punct = string.punctuation.replace('-', '').replace("'",'')
# regex to match intra-word dashes and intra-word apostrophes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")

#################
### functions ###
#################

# returns the vector of a word
def my_vector_getter(word, wv):
    try:
		# we use reshape because cosine similarity in sklearn now works only for multidimensional arrays
        word_array = wv[word].reshape(1,-1)
        return (word_array)
    except KeyError:
        print 'word: <', word, '> not in vocabulary!'

# returns cosine similarity between two word vectors
def my_cos_similarity(word1, word2, wv):
    sim = cosine(my_vector_getter(word1, wv),my_vector_getter(word2, wv)) 
    return (round(sim, 4))

# plots word vectors
def plot_points(my_names, my_wv, dims=(1,2)):
    
    my_vectors = [my_vector_getter(elt, wv=my_wv) for elt in my_names]
    dim_1_coords = [element[0,dims[0]-1] for element in my_vectors]
    dim_2_coords = [element[0,dims[1]-1] for element in my_vectors]
    	
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dim_1_coords, dim_2_coords, 'ro')
    diff1 = max(dim_1_coords) - min(dim_1_coords)
    diff2 = max(dim_2_coords) - min(dim_2_coords)
    plt.axis([min(dim_1_coords)-0.1*diff1, max(dim_1_coords)+0.1*diff1, min(dim_2_coords)-0.1*diff2, max(dim_2_coords)+0.1*diff2])
    
    for x, y, name in zip(dim_1_coords , dim_2_coords, my_names):     
        ax.annotate(name, xy=(x, y))
    	
    plt.grid()
    plt.show()

# performs basic pre-processing
# note: we do not lowercase for consistency with Google News embeddings
def clean_string(string, punct=punct, my_regex=my_regex):
    # remove formatting
    str = re.sub('\s+', ' ', string)
	# remove punctuation (preserving dashes)
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str

# to fill
#path_to_stopwords = 
	
with open(path_to_stopwords + 'smart_stopwords.txt', 'r') as my_file: 
    stpwds = my_file.read().splitlines()

newsgroups = fetch_20newsgroups()
docs, labels = newsgroups.data, newsgroups.target

lists_of_tokens = []

for i, doc in enumerate(docs):
    # clean document with the clean_string() function
    #### your code here ####
    # tokenize (split based on whitespace)
    tokens = doc.split(' ')
    # remove stopwords
    tokens = [token for token in tokens if token not in stpwds]
    # remove tokens less than 2 characters in size
    tokens = [token for token in tokens if len(token)>=2]
    # save result
    lists_of_tokens.append(tokens)
    if i%1e3 == 0:
        print i, 'docs processed'

# save processed documents to disk
# to fill
#path_to_data = 
name_persist = 'processed_docs'
d = shelve.open(path_to_data + name_persist)
d['lot'] = lists_of_tokens
d.close()

docs = [' '.join(list) for list in lists_of_tokens]

# create empty word vectors for the words in vocabulary	
# we set size=300 to match dim of GNews word vectors
mcount = 5
vectors = Word2Vec(size=3e2, min_count=mcount)

# build vocabulary for 'vectors' from the list of lists of tokens with the build_vocab() method
#### your code here ####

vocab = [elt[0] for elt in vectors.vocab.items()]

all_tokens = [token for sublist in lists_of_tokens for token in sublist]

t_counts = dict(Counter(all_tokens))

# sanity check (should return True)
len(vocab) == len([token for token, count in t_counts.iteritems() if count>=mcount])

# replace our empty word vectors with GNews ones
# to fill
#path_to_wv =

# we load only the Google word vectors corresponding to our vocabulary
vectors.intersect_word2vec_format(path_to_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

######################################
# experimenting with word embeddings #
######################################

# two similar words
my_cos_similarity('man','woman', wv=vectors)

# two dissimilar words
my_cos_similarity('man','paperwork', wv=vectors)

# examples of concepts captured in the embedding space:

# country-capital
France = my_vector_getter('France', wv=vectors)
Paris = my_vector_getter('Paris', wv=vectors)
Germany = my_vector_getter('Germany', wv=vectors)
Berlin = my_vector_getter('Berlin', wv=vectors)

operation = France - Paris + Berlin
round(cosine(operation, Germany),5)

# gender
# repeat the steps above for 'man', 'woman', 'king', 'queen'
#### your code here ####

# we will visualize regularities among word vectors in a 2 dimensional space

# to preserve mapping, create (1) an array with the vectors of the words in vocab, (2) a list containg the actual words

vectors_array = np.empty(shape=vectors.syn0.shape)
mapping = []
for i, elt in enumerate(vectors.vocab.items()):
    word = elt[0]
    mapping.append(word)
    vectors_array[i,] = vectors[word]

my_pca = PCA(n_components=4)

wv_2d_values = my_pca.fit_transform(vectors_array) 

# finally, create dictionary with the words in vocab as keys and the 2-dimensional projections as values
wv_2d = {}
for i, word in enumerate(mapping):
    wv_2d[word] =  wv_2d_values[i,]

words_subset = ['France','Paris','Italy','Rome','Spain','Madrid','Ankara','Turkey']
plot_points(my_names=words_subset, my_wv=wv_2d)

s_1 = 'Kennedy was shot dead in Dallas'
s_2 = 'The President was killed in Texas'

# compute the features of the vector space (unique non-stopwords)

# tokenize and remove stopwords, store in s_1 and s_2
#### your code here ####

# the features are all the unique remaining words
features = list(set(s_1).union(set(s_2)))

# project the two sentences in the vector space
p_1 = [1 if feature in s_1 else 0 for feature in features]
p_1_bow = zip(features, p_1)

# repeat same steps for the second sentence
#### your code here ####

print "representation of '", s_1, "' : \n",
print p_1_bow

print "representation of '", s_2, "' : \n",
print p_2_bow

# 1) compute the similarity of these two sentences in the vector space
# what do you observe?

round(cosine(np.array(p_1).reshape(1,-1),np.array(p_2).reshape(1,-1)), 5)

# now, if we use the word embedding space

p_1_embeddings = np.concatenate([my_vector_getter(word, vectors) for word in s_1])
# compute centroid
centroid_1 = np.mean(p_1_embeddings, axis=0).reshape(1,-1)

# repeat steps above for the second sentence
#### your code here ####

# 2) compute cosine similarity between sentence centroids
# this time we can see that the semantic similarity between the two sentences is captured
round(cosine(centroid_1, centroid_2),5)

# we can even try with the Word Mover's Distance

# should be small
vectors.wmdistance(s_1,s_2)

# should be null
vectors.wmdistance(s_1,s_1)

# we can also try with different sentences:
s_1 = 'Obama speaks to the media in Illinois'
s_2 = 'The President addresses the press in Chicago'
s_1 = [word for word in s_1.split(' ') if word.lower() not in stpwds]
s_2 = [word for word in s_2.split(' ') if word.lower() not in stpwds]
vectors.wmdistance(s_1,s_2)

# compare with a completely different sentence
# we can see that it is higher
s_3 = 'not all computer science students are geeks'
s_3 = [word for word in s_3.split(' ') if word.lower() not in stpwds]
vectors.wmdistance(s_1,s_3)
vectors.wmdistance(s_2,s_3)

#######################################
### unsupervised doc classification ###
#######################################

# read results generated by the 'k_fold_cv.py' script

# to fill
#ath_to_results = 
name_persist = 'my_results'

d = shelve.open(path_to_results + name_persist)
all_folds = d['results']
d.close()

name_persist ='folds_data'
d = shelve.open(path_to_results + name_persist)
folds_data = d['folds_data']
d.close()

n_folds = 4
ks = [1,3,5,7,11,17]

# predictions and solutions (integer lists of length the number of documents used)
all_preds = []
all_sols = []
for i in range(n_folds):
    all_sols = all_sols + folds_data[i]['labels_test']
    all_preds = all_preds + all_folds[i]

preds_fold_tfidf = [elt['tfidf'] for elt in all_preds]

for idx, k in enumerate(ks):
    print 'accuracy for k=',k,':', #### your code here #### - use 'accuracy_score' scikit learn function 

preds_fold_wmd = [elt['wmd'] for elt in all_preds]

for idx, k in enumerate(ks):
    print 'accuracy for k=',k,':', #### your code here #### - use 'accuracy_score' scikit learn function 