import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def casefolding(text):
  text = text.lower()
  text = re.sub(r'[@,.?#]', '', text)                               
  text = re.sub(r'https?://\S+|www\.\S+', '', text) 
  text = re.sub(r'[-+]?[0-9]+', '', text)           
  text = re.sub(r'[^\w\s]','', text)                
  text = text.strip()
  return text

def text_normalize(text,key_norm):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
  text = str.lower(text)
  return text

more_stopword = ['tsel', 'gb', 'rb', 'ttg', 'ga', 'nih', 'bgt', 'ni', 'nya']

def remove_stop_words(text,stopwords_ind):
  clean_words = []
  text = text.split()
  for word in text:
      if word not in stopwords_ind:
          clean_words.append(word)
  return " ".join(clean_words)

def stemming(text,stemmer):
  text = stemmer.stem(text)
  return text

def text_preprocessing_process(text,key_norm,stopwords_ind,stemmer):
  text = casefolding(text)
  text = text_normalize(text,key_norm)
  text = remove_stop_words(text,stopwords_ind)
  text = stemming(text,stemmer)
  return text