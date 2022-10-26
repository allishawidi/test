'''
	Contoh Deloyment untuk Domain Natural Language Processing (NLP)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from joblib import load
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from fungsi import *

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

stopwords_ind = None
key_norm      = None
factory       = None
stemmer       = None
vocab         = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]		
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	text_input = ""
	
	if request.method=='POST':
		text_input = request.form['data']
		
		text_input = text_preprocessing_process(text_input,key_norm,stopwords_ind,stemmer)

		tf_idf_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(vocab))

		hasil = model.predict(tf_idf_vec.fit_transform([text_input]))

		if(hasil==0):
			hasil_prediksi = "Normal"
		elif (hasil==1):
			hasil_prediksi = "Penipuan"
		else:
			hasil_prediksi = "Promo"
		
		return jsonify({
			"data": hasil_prediksi,
		})

# =[Main]========================================

if __name__ == '__main__':
	
	# Setup
	stopwords_ind = stopwords.words('indonesian')
	stopwords_ind = stopwords_ind + more_stopword
	
	key_norm = pd.read_csv('key_norm.csv')
 
 	# key_norm2 = pd.read_csv('key_norm2.csv')
  
  	# data_gabungan = key_norm.append(key_norm2, ignore_index=True)
	
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	
	vocab = pickle.load(open('kbest_feature.pickle', 'rb'))
	
	model = load('model_spam_tfidf_nb.model')

	app.run(host="localhost", port=5000, debug=True)
	
	


