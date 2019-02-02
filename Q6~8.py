# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:09:55 2019

@author: superray
"""
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.pipeline import Pipeline
import numpy as np 
np.random.seed(42) 
import random 
random.seed(42)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn import metrics 
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token.strip(string.punctuation) for token in tokens if token.isalnum()]
    return tokens

def contingency_table(true_labels, pre_labels):
    n_clusters = len(np.unique(pre_labels))
    A = np.zeros(shape = (n_clusters,n_clusters))
    uniq_true = np.unique(true_labels)
    for i, true_label in enumerate(uniq_true):
        for j, pre_label in enumerate(pre_labels):
            if(true_labels[j] == true_label):
                A[i][pre_label] += 1
    return A

def evaluate(labels, km_labels_):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km_labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km_labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km_labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km_labels_))
    print("Adjusted Mutual Information Score: %.3f" % metrics.adjusted_mutual_info_score(labels, km_labels_))
# =============================================================================
#     print("Contingency Table: ")
#     A = contingency_table(labels, km_labels_).astype(int)
#     if len(np.unique(labels)) > 2:
#         print(plt.matshow(A, cmap=get_cmap('Blues')))
#     else:
#         print(A)
# =============================================================================


def visualization(data_matrix,km_labels_,gt = True):
    plt.scatter(data_matrix[:,0],data_matrix[:,1],c = km_labels_)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    if gt:
        plt.title('Clustering results-Ground truth')
    else:
        plt.title('Clustering results-Kmean')
    plt.show()
    
categories = ['comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware','rec.autos',
'rec.motorcycles','rec.sport.baseball',
'rec.sport.hockey']


dataset = fetch_20newsgroups(subset = 'all', categories = categories, shuffle = True, random_state = 42)

vfunc = np.vectorize(lambda t : int(t / 4))
labels = vfunc(dataset.target)

tfidfVectorizer = TfidfVectorizer(min_df=3, stop_words='english', tokenizer = tokenize)
nmf = NMF(n_components=2, init='random', random_state=0)
kmeans = KMeans(n_clusters = 2, random_state = 0, max_iter = 1000, n_init = 30)
scaler = StandardScaler()

'''scal with SVD only'''
pp1 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', TruncatedSVD(n_components=3, random_state=0)),
    ('scal',scaler),
])

pro1data = pp1.fit_transform(dataset.data)
klabels = kmeans.fit_predict(pro1data)
evaluate(labels, klabels)

visualization(pro1data,klabels,False)
visualization(pro1data,labels)

'''log with SVD only'''
pp2 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', TruncatedSVD(n_components=3, random_state=0)),
])
c = 0.01
pro1data = pp2.fit_transform(dataset.data)

signM = np.sign(pro1data)
logtransform = np.log(np.abs(pro1data)+c)-np.log(c)
pro2data = np.multiply(signM,logtransform)

klabels = kmeans.fit_predict(pro2data)
evaluate(labels, klabels)
visualization(pro2data,klabels,False)
visualization(pro2data,labels)

'''scal and log with SVD'''
pp3 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', TruncatedSVD(n_components=3, random_state=0)),
    ('scal',scaler),
])

pro1data = pp3.fit_transform(dataset.data)
signM = np.sign(pro1data)
logtransform = np.log(np.abs(pro1data)+c)-np.log(c)
pro2data = np.multiply(signM,logtransform)

klabels = kmeans.fit_predict(pro2data)
evaluate(labels, klabels)
visualization(pro2data,klabels,False)
visualization(pro2data,labels)

'''log and scale with SVD'''
pp4 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', TruncatedSVD(n_components=3, random_state=0)),
])

pro1data = pp4.fit_transform(dataset.data)
signM = np.sign(pro1data)
logtransform = np.log(np.abs(pro1data)+c)-np.log(c)
pro2data = np.multiply(signM,logtransform)

pro3data = scaler.fit_transform(pro2data)

klabels = kmeans.fit_predict(pro3data)
evaluate(labels, klabels)
visualization(pro3data,klabels,False)
visualization(pro3data,labels)

'''scal with NMF only'''
pp1 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', nmf),
    ('scal',scaler),
])

pro1data = pp1.fit_transform(dataset.data)
klabels = kmeans.fit_predict(pro1data)
evaluate(labels, klabels)
visualization(pro1data,klabels,False)
visualization(pro1data,labels)


'''log with NMF only'''
pp2 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', nmf),
])
c = 0.01
pro1data = pp2.fit_transform(dataset.data)

signM = np.sign(pro1data)
logtransform = np.log(np.abs(pro1data)+c)-np.log(c)
pro2data = np.multiply(signM,logtransform)

klabels = kmeans.fit_predict(pro2data)
evaluate(labels, klabels)
visualization(pro2data,klabels,False)
visualization(pro2data,labels)

'''scal and log with NMF'''
pp3 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim', nmf),
    ('scal',scaler),
])

pro1data = pp3.fit_transform(dataset.data)
signM = np.sign(pro1data)
logtransform = np.log(np.abs(pro1data)+c)-np.log(c)
pro2data = np.multiply(signM,logtransform)

klabels = kmeans.fit_predict(pro2data)
evaluate(labels, klabels)
visualization(pro2data,klabels,False)
visualization(pro2data,labels)

'''log and scale with NMF'''
pp4 = Pipeline([
    ('vect',tfidfVectorizer),
    ('reduce_dim',nmf),
])

pro1data = pp4.fit_transform(dataset.data)
signM = np.sign(pro1data)
logtransform = np.log(np.abs(pro1data)+c)-np.log(c)
pro2data = np.multiply(signM,logtransform)

pro3data = scaler.fit_transform(pro2data)

klabels = kmeans.fit_predict(pro3data)
evaluate(labels, klabels)
visualization(pro3data,klabels,False)
visualization(pro3data,labels)