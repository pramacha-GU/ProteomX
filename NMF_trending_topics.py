'''
This program aims at finding and plotting the trending topics over the years.
Non-negative matrix factorization (NMF) was used to rank articles into various
topics. A plot of articles published over the years in various topics was also generated
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def unpickle_abstract_Series(abstract_series):
    '''
    Cleaned up abstract lst was pickled for future use. Here I unpickle the list
    '''
    abstract_doc = pd.read_pickle(abstract_series)
    return pd.Series(abstract_doc)

def tfidf_vec(abstract_word_documents):
    '''
    Converting documents into a TFIDF vector. Retaining only 2000 features
    '''
    vectorizer = TfidfVectorizer(max_df=0.5, max_features = 2000, \
                             min_df=2, stop_words='english')
    return vectorizer.fit_transform(abstract_word_documents).toarray()

def nmf_WH(X_vec):
    '''
    Non-negative matrix factorization of the TFIDF vector.
    W and H vectors were also pickled
    '''
    nmf = NMF(n_components=5, random_state=1,alpha=.1, l1_ratio=.5)
    W_sklearn = nmf.fit_transform(X_vec)
    H_sklearn = nmf.components_
    with open('W_sklearn.pkl', 'w') as f:
        pickle.dump(W_sklearn, f)
    with open('H_sklearn.pkl', 'w') as f:
        pickle.dump(H_sklearn, f)
    return W_sklearn, H_sklearn

def get_topic(W_mat):
    '''
    This function goes through the W matrix and ranks the articles into possible
    topics
    '''
    abs_list = []
    for x,y in enumerate(W_mat):
        abs_list.append((np.argsort(y))[::-1])
    return abs_list

def topic_assign_abstract(abs_list):
    '''
    This function goes through the topics ranks for articles and assigns them to
    one topic
    '''
    latent_feature_rank = []
    for i in abs_list:
        latent_feature_rank.append(i[0]+1)
    return latent_feature_rank

def add_nmf_rank(nmf_rank):
    '''
    Adding the NMF rank to the pubmed_cleaned document
    '''
    df = pd.read_csv('pubmed_cleaned.txt', sep='|')
    df['Topic_num'] = pd.Series(nmf_rank)
    df.to_csv('pubmed_cleaned.txt', index=False, sep='|')

def rank_year_group(df_month_clus):
    '''
    Plotting the number of articles in each topic ober the years
    '''
    yr = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
    dfmyN_yrcount_list = [[],[],[],[],[]]
    for topcs in xrange(5):
        for i1 in yr:
            try:
                dfmyN_yrcount_list[topcs].append(((dfmyN[dfmyN[2]==topcs+1]).groupby([0,1]).count()).ix[i1,:].sum()[2])
            except KeyError:
                dfmyN_yrcount_list[topcs].append(0)
    fig = plt.figure(figsize = (10,8))
    plt.plot(np.array(dfmyN_yrcount_list[0]), color='blue', lw=2)
    plt.plot(np.array(dfmyN_yrcount_list[1]), color='green', lw=2)
    plt.plot(np.array(dfmyN_yrcount_list[2]), color='red', lw=2)
    plt.plot(np.array(dfmyN_yrcount_list[3]), color='purple', lw=2)
    plt.plot(np.array(dfmyN_yrcount_list[4]), color='black', lw=2)
    plt.xlabel('Year')
    plt.ylabel('Number of publications')
    fig.savefig('topic_yr.jpg')

if __name__ == '__main__':
    abstract_series = unpickle_abstract_Series('cleaned_abstract_list.pkl')
    X_vec = tfidf_vec(abstract_series)
    vec_feature_words = X_vec.get_feature_names()
    W_sklearn_T, H_sklearn = nmf_WH(X_vec)
    abs_list = get_top_abstracts(W_sklearn.T, range(len(abstract_series)), len(abstract_series))
    nmf_rank = topic_assign_abstract(abs_list)
    add_nmf_rank(nmf_rank)
    df = pd.read_csv('pubmed_cleaned.txt', sep='|')
    rank_year_group(df[['Year', 'Month', 'NMF_cluster']])
