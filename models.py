from django.db import models

# Create your models here.
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import requests
from bs4 import BeautifulSoup
import re   
import nltk
nltk.download('punkt')   # one time execution
from nltk.tokenize import sent_tokenize
import re
nltk.download('stopwords')  # one time execution
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cosine


class extractive:
  def __init__(self,topic):
      self.topic = topic
      self.data = ""
      self.sum = ""
      try:
        self.scrapflip()
        self.preprocessing()
        self.Tf_idf()
        self.Kmeans()
        self.csimilarity()
        self.Print()
      except:
        self.sum = "Data couldn't be found"

      
      

  def scrapflip(self):
      Url = 'https://en.wikipedia.org/wiki'
      url = Url + '/' + self.topic
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')      
      for para in soup.find_all('p'):
          self.data += re.sub(r'\[[0-9]*\]', r'', para.get_text())

  def preprocessing(self):
    self.sentence = sent_tokenize(self.data)
    self.corpus = []
    for i in range(len(self.sentence)):
      sen = re.sub('[^a-zA-Z]', " ", self.sentence[i])  
      sen = sen.lower()                            
      sen = sen.split()                         
      sen = ' '.join([i for i in sen if i not in stopwords.words('english')])   
      self.corpus.append(sen)
    self.wd = []
    for sent in self.corpus:    
      for s in sent.split():
        try:
          p = self.wd.index(s)
        except:   
          self.wd.append(s)
    self.wd.sort()


  def getWordFreqs(self,sen):
    wordFreqs = {}
    for word in sen.split():
        if word not in wordFreqs.keys():
            wordFreqs[word] = 1
        else:
            wordFreqs[word] = wordFreqs[word] + 1
    return wordFreqs
  
    
    
  def Tfs(self,sentences):
    wc = []    
    for sent in sentences:
      sc = []
      wordFreqs = self.getWordFreqs(sent)        
      for word in self.wd:
        if wordFreqs.get(word, 0) != 0:
            sc.append(wordFreqs[word])
        else:
            sc.append(0)    
      wc.append(sc)
    return wc 
    
  def IDFs(self,sentences):
        N = len(sentences)
        idfs = {}
        words = {}
        w2 = []
        for sent in sentences:
            for word in sent.split():
                if self.getWordFreqs(sent).get(word, 0) != 0:
                    words[word] = words.get(word, 0) + 1
        for word in words:
            n = words[word]
            try:
                w2.append(n)
                idf = math.log10(float(N) / n)
            except ZeroDivisionError:
                idf = 0
            idfs[word] = idf
        return idfs
                                       
  def Tf_idf(self):
    tfs = self.Tfs(self.corpus)
    #print(tfs)
    idfs = self.IDFs(self.corpus)
    self.tfidf = tfs
    for i in range(len(self.tfidf)):
        ct = 0
        for word in self.wd:
            tf_ifs = idfs[word]*self.tfidf[i][ct]
            self.tfidf[i][ct]=tf_ifs
            ct+=1


  def Kmeans(self):
    self.X = np.array(self.tfidf)
    self.n_clusters = 5
    self.kmeans = KMeans(self.n_clusters, init = 'k-means++', random_state = 42)
    self.y_kmeans = self.kmeans.fit_predict(self.X)
    #print(self.kmeans.cluster_centers_)
    #print(self.kmeans.labels_)
    #plt.scatter(self.X[:,0], self.X[:,1], c=self.kmeans.labels_, cmap='rainbow')
    #plt.scatter(self.kmeans.cluster_centers_[:,0] ,self.kmeans.cluster_centers_[:,1], color='black')
    self.closest, _ = pairwise_distances_argmin_min(self.kmeans.cluster_centers_, self.X)   

  def similarity(self,v1, v2):
        score = 0.0
        if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
            score = ((1 - cosine(v1, v2)) + 1) / 2
        return score

  def csimilarity(self):             
    my_list=[]
    sentences_scores = []
    for i in range(self.n_clusters):   
        my_clus={}
        for j in range(len(self.y_kmeans)):        
            if self.y_kmeans[j]==i:
                score = self.similarity(self.X[j, :], self.X[self.closest[i],:])
                sentences_scores.append((j, self.sentence[i], score, self.X[j, :]))
                my_clus[j]=score
        my_list.append(sorted(my_clus))        
    self.sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)

  def Print(self):
    sentences_summary = []
    results = []
    slist = []
    self.sum=""
    for s in self.sentence_scores_sort:
        if len(sentences_summary) == 20:
            break
        include_flag = True
        for ps in sentences_summary:
            sim = self.similarity(s[3], ps[3])
            if sim > .7:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            slist.append(s[0])
    for i in sorted(slist):
        self.sum = self.sum +"\n" +self.sentence[i]
