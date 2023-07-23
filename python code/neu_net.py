'''
with the present we'll train the text data along with the financial data and get our trained net.
this will require some manipulation of the current state of the data, which makes sense that is done here,
in order to promote homogeneity and peace of mind...
'''
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

#
# Fetch articles data
with open('E:/Tralgo/articles/article_container.json', 'r') as f:
    articles_text = json.load(f)

#
# in this part I worry about getting a word tokenizer going

flat_articles_list = [item for sublist in articles_text.values() for item in sublist]

#
# we tokenize our text data with their relevant tags from the dictionary

tagged_data = []
for period, articles in articles_text.items():
    for article in articles:
        tag = period
        tagged_data.append(TaggedDocument(words=article.lower().split(), tags=[tag]))

#
# here we train our Doc2Vec model on every word of every article with attention to tags
doc2vec_model = Doc2Vec(tagged_data, vector_size=300, window=5, min_count=1, workers=4)


# in this part we transform words into vectors
vect_dict = dict()
vectorizer = TfidfVectorizer()
for key in articles_text:
    articles = articles_text[key]
    vectors = vectorizer.fit_transform(articles)

    # convert to PyTorch tensors
    vectors_np = vectors.toarray()
    vectors_tensor = torch.tensor(vectors_np, dtype=torch.float32)

    vect_dict[key] = vectors_tensor


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