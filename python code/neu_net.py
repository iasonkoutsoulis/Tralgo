'''
with the present we'll train the text data along with the financial data and get our trained net.
this will require some manipulation of the current state of the data, which makes sense that is done here,
in order to promote homogeneity and peace of mind...
'''
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

with open('E:/Tralgo/articles/articles_of_2009-7-B2.txt') as f:
    article_text = f.readlines()

vectorizer = TfidfVectorizer()
X = vectorizer.fit(article_text)

print(X.vocabulary_)
print(X.idf_)

vector = X.transform([article_text[0]])
print(vector.shape)
print(vector.toarray())




# a = torch.ones(3)
# float(a[1])
# a[2] = 2.0

# points = torch.tensor([[4.0, 1.0], [5.0, 3.0]]) # tensor takes only one positional argument
# points.shape
# points[1, 1]

# img_t = torch.randn(3, 5, 5)
# weights = torch.tensor([0.2126, 0.7152, 0.0722])
# batch_t = torch.randn(2, 3, 5, 5) # (batch, channels, rows, columns)

# weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])