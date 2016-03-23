import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import gensim

# this document contains pubmed records
df = pd.read_csv('data/pubmed_master.txt', sep='|')
#Word frequency vector for the abstract documents
word_freq = pd.read_csv('data/word_freq_vec1.csv', header=None)
#word_features for the word frequency vector
vec_feature_words = pd.read_pickle('data/vec_feature_words_2500.pkl')
#word2vec model to be used
w2v_model = gensim.models.Word2Vec.load('data/w2vmodel')
#convert features to string
for i,j in enumerate(vec_feature_words):
	try:
		vec_feature_words[i]=str(j)
	except UnicodeEncodeError:
		pass


from flask import Flask, request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('PR_webapp.html')
#http://localhost:6969/process_html_data
@app.route('/process_html_data',methods=['GET','POST'])
def process_html_data():
	#get topics and keywords from the HTML form
	topic = request.form['topics']
	keywrd = request.form['keyword']
	if topic == 'all_db':
	    df1=df.copy()
	else:
		if topic == 'topic_1':
			df1 = df[df['NMF_cluster']==1]
		if topic == 'topic_2':
			df1 = df[df['NMF_cluster']==2]
		if topic == 'topic_3':
			df1 = df[df['NMF_cluster']==3]
		if topic == 'topic_4':
			df1 = df[df['NMF_cluster']==4]
		if topic == 'topic_5':
			df1 = df[df['NMF_cluster']==5]
	#if length of the keyword is more than 1, go ahead otherwise ask for a keyword
	if len(keywrd)>0:
		keywrd_dict = {}
		keywrd_list = keywrd.lower().split()
		#if entered keyword is in the feature list, give it a score of 1. Otherwise 0
		for kw in keywrd_list:
			if kw in vec_feature_words:
				keywrd_dict[kw] = 1.0
			else:
				keywrd_dict[kw] = 0.0
			o_keywrd_keys = keywrd_dict.keys()
			#using word2vec model get similar words. Keep if similarity exceeds 0.8
			for j in o_keywrd_keys:
				try:
					similar_words = w2v_model.most_similar(positive=[j], topn=10)
				except KeyError:
					similar_words = ''
				for k in similar_words:
					if k[0] in vec_feature_words:
						keywrd_dict[k[0]] = k[1]
		keywrd_items = keywrd_dict.items()
		feature_index = []
		thresh = 0.8
		len_findx = 0
		while len_findx==0:
			for i in keywrd_items:
				if i[1]>thresh:
					feature_index.append(vec_feature_words.index(i[0]))
				len_findx = len(feature_index)
				if len_findx==0:
					thresh-=0.1
				if thresh<=0.0:
					data1=['No records found. Please specify another search term']
					len_findx=1
		print 'good so far2'
		if len(feature_index)>0:
			if topic=='all_db':
				wfarray = np.array(word_freq.ix[df1.index,feature_index])
			else:
				wfarray = np.array(word_freq.ix[df1.index,feature_index])
			wfarray.astype(float)
			ifw_wf = []
			#print 'good so far2a'
			if topic=='all_db':
				ifw_arr = df1['Impact_Factor']
			else:
				ifw_arr = df1['Impact_Factor']
			ifw_arr = ifw_arr.reshape(len(wfarray),1)
			for i,j in enumerate(wfarray):
				ifw_wf.append(np.sum(wfarray[i]*ifw_arr[i]))
			ifw_wf = pd.DataFrame(ifw_wf)
			ifw_wf['auth_aff_unq_id']= df1['auth_aff_unq_id']
			ifw_wf1 = ifw_wf.groupby('auth_aff_unq_id', as_index=False).sum()
			ifw_wf_sorted = ifw_wf1.sort_values(by=0, ascending=0)
			ifw_wf_sorted_in  =ifw_wf_sorted['auth_aff_unq_id'].index
			top_10_auth = ifw_wf_sorted.ix[ifw_wf_sorted_in[:10],]['auth_aff_unq_id']
			data1 = []
			#print 'good so far2b'
			for i in top_10_auth:
				data1.append(list(df1[df1['auth_aff_unq_id']==i][['Last_Author','Last_Auth_Affns']].values[0]))
			#print 'good so far3'
			for i, j in enumerate(data1):
				data1[i] = ' : '.join(j)
		else:
			data1=['No records found. Please specify another search term']
		#print 'good so far4', data1
	else:
		data1=['No records found. Please specify another search term']
	for i,j in enumerate(data1):
		try:
			data1[i] = j.encode("utf-8")
		except:
			print j
			data1[i] = ''
	return render_template('PR_webapp.html', data=data1)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
