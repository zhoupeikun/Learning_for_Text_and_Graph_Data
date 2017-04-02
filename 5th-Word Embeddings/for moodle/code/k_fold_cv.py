###################
#### LIBRARIES ####
###################

import re
import time
import string
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from gensim.models.word2vec import Word2Vec
from multiprocessing import Pool, cpu_count
from functools import partial
import shelve

print 'packages loaded'

###################
#### FUNCTIONS ####
###################

def majority_voting(labels,sorted_index,ks):
    preds = []
    for k_nn in ks:
        # get labels of the 'k_nn' nearest neighbors
        nn_labels = [labels[i] for i in sorted_index][:k_nn]
        # get most represented label and use it as the prediction
        counts = dict(Counter(nn_labels))
        #### your code here #### store the max value in the 'counts' dictionary as 'max_counts'
        pred = [k for k,v in counts.iteritems() if v==max_counts][0]
        preds.append(pred)
    return preds

def instance_predict(instance, collection, labels, vectors, vect, doc_term_mtx, ks):
    
    """ 
    - predicts the label of a new instance ('instance') with a Knn approach w.r.t. a collection ('collection') in terms of both the WMD and cosine similarity with TFIDF vectors for various values of K (stored in the 'ks' list) 
    - returns a dictionary with two keys (names of the two methods) and lists of length len(ks) as values. Each list contains the predictions of the method for each value of K	
	"""
    
    #### COMPUTE COSINE SIMILARITY WITH TFIDF VECTORS ####	
    	
    # get tfidf vector of the instance (using vocab from training set only - hence the 'transform' only (no fit))
    instance_tfidf = vect.transform([instance[0]])
    # compute cosine similarity between new instance and all elements in the collection
    sims = cosine(doc_term_mtx, Y=instance_tfidf, dense_output=True).tolist()
    sims = [elt[0] for elt in sims]
    # get indexes of elements sorted by DECREASING order (the greater the better for cosine sim) and store them in 'idx_st_cos'
    #### your code here ####
    
    #### COMPUTE WMD ####
    
    # wmdistance accepts lists of tokens (2nd entry of each tuple)
    dists = [vectors.wmdistance(instance[1],tuple[1]) for tuple in collection]
    # get indexes of elements sorted by INCREASING order (the smaller the better for the WMD) and store them in 'idx_st_wmd'
    #### your code here ####	
    
    #### GENERATE PREDICTIONS ####
    
    predictions = {}
    predictions['tfidf'] = #### your code here #### use the 'majority_voting' function
    predictions['wmd'] = #### your code here #### use the 'majority_voting' function
        
    return predictions	

# we will map 'do_one_fold' over 'folds_data'
def do_one_fold(one_fold_data, vectors, ks):
    """ 
    returns a list of dictionaries of length the number of observation in the test fold
	each dict contains the label predictions of an observation for each method ('tfidf' and 'wmd') and the values of nns (ks)
    """
    instances_train = one_fold_data['instances_train']
    labels_train = one_fold_data['labels_train']
    instances_test = one_fold_data['instances_test']
    
    collection = instances_train
    	
    # initialize vectorizer (we set the stop_words and preprocessor parameters to None since the docs have already been pre-processed)
    vect = TfidfVectorizer(min_df=5, stop_words=None, lowercase=False, preprocessor=None)
    
	# tfidf_vectorizer accepts raw docs (1st entry of each tuple)
    doc_term_mtx = vect.fit_transform([tuple[0] for tuple in collection])
            
    # generate predictions for the instances in the test fold
    predictions_all_instances = []
    
    counter = 0
    for instance in instances_test:
        # we get a dict with 2 keys ('tfidf' and 'WMD') containing each a list with the predictions for each value of K
        preds_for_inst = instance_predict(instance, collection, labels_train, vectors, vect, doc_term_mtx, ks)
        
        predictions_all_instances.append(preds_for_inst)
        
        counter += 1
        
    return predictions_all_instances

print 'functions defined'

##############
#### MAIN ####
##############

def main():
    # to fill    
    #path_to_stopwords =
    	
    with open(path_to_stopwords + 'smart_stopwords.txt', 'r') as my_file: 
        stpwds = my_file.read().splitlines()
    
    print 'stopwords loaded'	
    
    newsgroups = fetch_20newsgroups()
    labels = newsgroups.target
    
    print 'labels loaded'	
    
    lists_of_tokens = []
    
    # load processed documents (the file 'processed_docs' is generated in the main file)
    # to fill
    #path_to_data =
    name_persist = 'processed_docs'
    d = shelve.open(path_to_data + name_persist)
    lists_of_tokens = d['lot'] 
    d.close()
    
    docs = [' '.join(list) for list in lists_of_tokens]
    
    print 'pre-processed documents loaded'
    
    if len(labels) == len(docs):
       print '1st sanity check passed'
    else:
        print '1st sanity check failed!'
    
    # initialize word2vec model with empty word vectors of dimension 300		
    # we set size=300 to match dim of GNews word vectors
    mcount = 5
    vectors = Word2Vec(size=3e2, min_count=mcount)
    
	# build vocabulary for our data
    vectors.build_vocab(lists_of_tokens)
    vocab = [elt[0] for elt in vectors.vocab.items()]
    
    all_tokens = [token for sublist in lists_of_tokens for token in sublist]
    t_counts = dict(Counter(all_tokens))
    
    # sanity check (should return True)
    if len(vocab) == len([token for token, count in t_counts.iteritems() if count>=mcount]):
        print '2nd sanity check passed'
    else:
        print '2nd sanity check failed!'
    
    # fill our empty word vectors with the ones from Google News
    # to fill
    #path_to_wv =
    vectors.intersect_word2vec_format(path_to_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
    
    # normalize the vectors
    vectors.init_sims(replace=True)
    
    print 'word vectors loaded and normalized'
    
    # select the first n_doc documents (they have already been shuffled)
	# the larger n_doc, the better
    n_doc = 60 # takes approx. 4 mins on a quad-core i7. 200 takes about 40 mins
    docs_subset = docs[:n_doc]
    lot_subset = lists_of_tokens[:n_doc]
    labels_subset = labels[:n_doc]
    
    instances = zip(docs_subset, lot_subset)
    	
    print 'total number of observations used:', len(instances)
    
    # to speed-up computation of WMD, reduce the length of the documents bigger than 'threshold' words to the first 'threshold' words.
    th = 300
    
    print 'limiting doc size to', th, 'words'	
    
    instances = [(' '.join(tuple[1][:th]),tuple[1][:th]) if len(tuple[1])>=th else tuple for tuple in instances]
      
    # Knn classification with n_folds-fold cross-validation
    n_folds = 4
 
    print 'K-fold cross validation with K=', n_folds 
    	
    # number of nearest neighbors to try
    # (we use odd numbers for tie breaking)
    ks = [1,3,5,7,11,17]
    
    print 'numbers of nearest neighbors to try:', ks
    		
    # compute indexes for the 4-fold cross validation
    # at each iteration, we will use 3 folds for 'training' and 1 for testing
    
    fold_size = int(round(len(instances)/float(n_folds)))
    
    print 'creating the indexes for', n_folds, 'folds of size', fold_size
    
    index_folds = []
    index_fold = []
    k = 0
    
    for i in range(len(instances)):
        index_fold.append(i)
        k += 1
        if k == fold_size:
            print k
            index_folds.append(index_fold)
            index_fold = []
            k = 0
    
    # will contain the instances and labels required for each fold
    # as a list of length 'n_folds' containing dictionaries of length 4
    
    folds_data = []
    
    for fold in range(n_folds):
        one_fold_data = {}
    
        training_fold_indexes = [elt for elt in range(n_folds) if elt != fold]
        training_indexes = [index_folds[idx] for idx in training_fold_indexes]
        # flatten list of lists into a list
        training_indexes = [idx for sublist in training_indexes for idx in sublist]
        one_fold_data['instances_train'] = [instances[idx] for idx in training_indexes]
        one_fold_data['labels_train'] = [labels[idx] for idx in training_indexes]
        
        # here, no flattening needed since we only select the elements from a single sublist
        test_indexes = index_folds[fold]
        one_fold_data['instances_test'] = [instances[idx] for idx in test_indexes]
        one_fold_data['labels_test'] = [labels[idx] for idx in test_indexes]
        
        folds_data.append(one_fold_data)
    
    # persist data to disk
    # to fill
    #path_to_results =
    name_persist = 'folds_data'
    d = shelve.open(path_to_results + name_persist)
    d['folds_data'] = folds_data
    d.close()
    
    print 'folds data ready and saved to disk'
    
	# we will map 'do_one_fold' over 'folds_data'
    # to this purpose, we create a partial version of 'do_one_fold' where all parameters are fixed except the first one
    do_one_fold_partial = partial(do_one_fold, vectors=vectors, ks=ks)
     
    n_jobs = min(cpu_count(),n_folds)
    
    print 'using', n_jobs, 'cores'
    t = time.time()
    
    pool = Pool(processes=n_jobs)
    predictions_all_folds = pool.map(do_one_fold_partial, folds_data)
    pool.close()
    pool.join()
    
    print 'done in', time.time() - t, 'seconds'
    
    # persist results to disk
    # to fill
    #path_to_results =
    name_persist = 'my_results'
    d = shelve.open(path_to_results + name_persist)
    d['results'] = predictions_all_folds
    d.close()
    print 'results saved to disk'
    
if __name__ == "__main__":
    main()