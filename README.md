# nlphelpers

This repository consist of some of the interfaces to handle several functionalities in NLP.

# Requirements

1. Download glove.840B.300d.txt

# Functionalities:

1. Sentence2Vec:

It converts a sentence into fixed dimension vector

Usage:


```
from models import Sentence2Vec

sentence = 'i love you'

dimension = 100

s2v = Sentence2Vec(sentence,dimension)

s2v_vec = s2v.fit()

```

2. Sentence Similarity

Calculates similarity between target sentence and other sentences

Usage:

```
target = 'He is a bad person'

corpus = ['He is very good person','He is nasty','He is generous','He is cruel','He is kind','He is a racist']

maximum_number = 4

from models import Similarity

simi = Similarity(target,corpus,maximum_number)

sim_sen = simi.compute_similarity()

print(sim_sen)

```
