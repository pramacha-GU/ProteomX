'''
This program aims at finding the number of topics that the scientific articles
naturally fall into. To do this abstracts were word tokenized. Stop words were
removed and then combined together into a pandas Series. The abstract Series were
CountVectorized into a work count vector. Singlular value decomposition (SVD) was
performed. The sigma matrix was used to determine how many topics the articles
naturally fall into.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

def word_toknze(txt_name):
	'''
	Title and abstract were combined into a single document.After removing
	stop words documents were word tokenized. And recombined into a pandas Series.
	The Series was also pickled for future work.
	'''
	df = pd.read_csv(txt_name, sep='|')
    title_list = list(df['Title'])
    abstract_list = list(df['Abstract'])
    for i in xrange(len(abstract_list)):
        abstract_list[i] = title_list[i] + abstract_list[i]
    abstract_series = pd.Series(abstract_list)
    abstract_word_documents = []
    for sentence in abstract_series:
    try:
        abstract_word_documents.append(word_tokenize(sentence.lower()))
    except UnicodeDecodeError:
        abstract_word_documents.append((sentence.lower()).split())
    stop_words = stopwords.words('english')
    stop_words.extend([';', ',', '.', ':'])
    abstract_doc = []
    for j in abstract_word_documents:
        l1 = []
        for w in j:
            if not w in stop_words:
                l1.append(w)
        abstract_doc.append(' '.join(l1))
    with open('cleaned_abstract_list.pkl', 'w') as f:
    	pickle.dump(abstract_list, f)
    return pd.Series(abstract_doc)

def SVD_scree(abstract_word_documents):
	'''
	Singular value decomposition was performed on word count vector of the
	abstract Series. SVD was employed to get a sigma matrix. A scree plot
	suggested between 5-7 topics as being the right range.
	'''
    vec = CountVectorizer(max_features = 2000)
    X_vec = vec.fit_transform(abstract_word_documents).toarray()
    s_matrix = np.linalg.svd(X_vec, compute_uv=0)
    Sig2= s_matrix**2
    eigvals = Sig2[:10] / np.cumsum(s)[-1]
    fig = plt.figure(figsize=(8,5))
    sing_vals = np.arange(10) + 1
    plt.plot(sing_vals,eigvals,'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    fig.savefig('scree.jpg')

if __name__ == '__main__':
    abstract_word_documents = word_toknze('pubmed_cleaned.txt')
    SVD_scree(abstract_word_documents)
