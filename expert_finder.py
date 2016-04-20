'''
This program converts abstracts into word frequency vector weighted by the impact
factor of the journal they came from.  this matrix is saved as a csv to be used
later. A word2vec model is also trained and saved for use in the final app.
Feature words are also pickled to be used later in the app.
'''
import pandas as pd
import numpy as np
import cPickle as pickle
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer

def unpickle_abstract_Series(abstract_series):
    '''
    Cleaned up abstract lst was pickled for future use. Here I unpickle the list
    '''
    abstract_doc = pd.read_pickle(abstract_series)
    return pd.Series(abstract_doc)

def train_word2vec(abstract_doc):
    '''
    In this function a word2vec model is trained on scientific abstracts to
    build a vocabulary of words that will be used later
    '''
    model = gensim.models.Word2Vec()
    model = gensim.models.Word2Vec(abstract_doc, min_count=7)
    model.save('data/w2vmodel')
    return model


def count_vec(abstract_word_documents):
    '''
    Converting documents into a word frequenct vector. Retaining only 2000 features
    '''
    vec = CountVectorizer(max_features = 2000)
    vec1 = vec.fit_transform(abstract_word_documents)
    return vec1.toarray(), vec.get_feature_names()

if __name__ == '__main__':
    abstract_series = unpickle_abstract_Series('data/cleaned_abstract_list.pkl')
    word2vec_model = train_word2vec(abstract_series)
    X_vec, vec_feature_words = count_vec(abstract_series)
    X_vec = X_vec.astype(float)
    df = pd.read_csv('data/pubmed_master.txt', sep='|')
    X_vec_df_ifw = convert_count_vec_if(X_vec, df['Impact_Factor'])
    X_vec_dfifw['Impact_Factor'] = df['Impact_Factor']
    X_vec_dfifw['NMF_cluster'] = df['NMF_cluster']
    '''
    Saving word frequency vector. Pickling feature words.
    '''
    X_vec_dfifw.to_csv('data/word_freq_vec1.csv', index=False, sep='|')
    with open('vec_feature_words_2000.pkl', 'w') as f:
        pickle.dump(vec_feature_words,f)
