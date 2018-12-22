import os
import re
import numpy as np 
import scipy as sp 
from scipy.spatial.distance import cosine

class Sentence2Vec():
	def __init__(self,corpus,dim):
		if(dim<=0):
			raise ValueError("Invalid dimension mentioned please mention>=0")
		if(dim>300):
			raise ValueError("Please mention dimension <= 300")
		if(isinstance(corpus,list)):
			self.type = 'list'
			self.corpus = corpus
			self.dim = dim
			self.vectors = self.load_vectors()
		elif(isinstance(corpus,str)):
			self.type = 'string'
			self.corpus = corpus
			self.dim = dim
			self.vectors = self.load_vectors()
		else:
			raise ValueError('The inputs must be list of sentence or a string')

	def load_vectors(self):
		embedding_vectors = {}
		with open('glove.840B.300d.txt','r',encoding='utf-8') as f:
			for line in f.readlines():
				vals = line.split()
				word = vals[0]
				vector = np.asarray(vals[1:self.dim+1],dtype='float32')
				embedding_vectors[word] = vector

			return embedding_vectors

	def get_vector(self,sentence):
		words = sentence.split()
		sentence_vec = np.zeros(self.dim)
		count = 0
		for word in words:
			if(word in self.vectors):
				sentence_vec+=self.vectors[word]
				count+=1
		if(count):
			return sentence_vec/count
		else:
			return sentence_vec


	def fit(self):
		if(self.type=='list'):
			total_length = len(self.corpus)
			sentence_vectors = np.zeros((total_length,self.dim))
			for i in range(total_length):
				sentence_vectors[i,:] = self.get_vector(self.corpus[i])
			return sentence_vectors
		else:
			return self.get_vector(self.corpus)



class Similarity():
	def __init__(self,target,corpus,number):
		if((isinstance(corpus,list))!=1):
			raise ValueError("sentences must be in a list")
		if(len(target)==0):
			raise ValueError("Target Sentence should not be null")
		if(number>len(corpus)):
			raise ValueError("Please Mention Maximum number to be <= "+ str(len(corpus)))
		self.target = target
		self.corpus = corpus
		self.number = number
		self.vectors = self.load_vectors()


	def load_vectors(self):
		embedding_vectors = {}
		with open('glove.840B.300d.txt','r',encoding='utf-8') as f:
			for line in f.readlines():
				vals = line.split()
				word = vals[0]
				vector = np.asarray(vals[1:],dtype='float32')
				embedding_vectors[word] = vector

			return embedding_vectors

	def get_vector(self,sentence):
		words = sentence.split()
		sentence_vec = np.zeros(300)
		count = 0
		for word in words:
			if(word in self.vectors):
				sentence_vec+=self.vectors[word]
				count+=1
		if(count):
			sentence_vec = sentence_vec/count
		return sentence_vec.reshape((1,300))


	def compute_similarity(self):
		total_length = len(self.corpus)
		self.target_vector = self.get_vector(self.target)
		similarity = []
		for i in range(total_length):
			self.corpus_vec = self.get_vector(self.corpus[i])
			similarity.append(((cosine(self.target_vector,self.corpus_vec)),self.corpus[i]))
		similarity_sorted =  sorted(similarity,key=lambda x:x[0],reverse=True)
		similar_sentences =[]
		for i in range(self.number):
			similar_sentences.append(similarity_sorted[i])

		return similar_sentences