'''
with the present we'll train the text data along with the financial data and get our trained net.
this will require some manipulation of the current state of the data, which makes sense that is done here,
in order to promote homogeneity and peace of mind...
'''
import pyarrow.feather as feather
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim


with open('E:/Tralgo/articles/article_container.json', 'r') as f:
    articles_text = json.load(f)

vect_dict = dict()
vectorizer = TfidfVectorizer()
for key in articles_text:
    articles = articles_text[key]
    X = vectorizer.fit(articles)
    vector = X.transform(articles)

    vect_dict[key] = vector



df_arts = pd.DataFrame.from_dict(vect_dict, orient='index')
df_fins = feather.read_feather('E:/Tralgo/data/financial_container.csv')

dft = df_fins.join(df_arts)
dft.to_clipboard()

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