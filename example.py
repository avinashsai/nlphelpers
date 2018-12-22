import sys
import os
import re

from models import Sentence2Vec,Similarity

sentence = 'i love you'
dimension = 100

s2v = Sentence2Vec(sentence,dimension)
s2v_vec = s2v.fit()

target = 'He is a bad person'
corpus = ['He is very good person','He is nasty','He is generous','He is cruel','He is kind','He is a racist']
maximum_number = 4

simi = Similarity(target,corpus,maximum_number)
sim_sen = simi.compute_similarity()
print(sim_sen)

