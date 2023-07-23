'''
with the present we'll train the text data along with the financial data and get our trained net.
this will require some manipulation of the current state of the data, which makes sense that is done here,
in order to promote homogeneity and peace of mind...
'''
from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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

tagged_arts = []
for period, articles in articles_text.items():
    for article in articles:
        tag = period
        tagged_arts.append(TaggedDocument(words=article.lower().split(), tags=[tag]))

#
# here we train our Doc2Vec model on every word of every article with attention to tags

doc2vec_model = Doc2Vec(tagged_arts, vector_size=300, min_count=2, epochs=10000)

#
# next we'll work on the embeddings a bit

doc_embeds = {}
for period, embeds in articles_text.items():
    doc_embeds[period] = doc2vec_model.docvecs[period]

doc_embeds_tens = {}
for period, embeds in doc_embeds.items():
    doc_embeds_tens[period] = np.array(embeds)

#
# in this part we transform words into vectors

vectorizer = TfidfVectorizer()
vectors = {period: vectorizer.fit_transform(articles).toarray().astype(np.float32).flatten() for period, articles in articles_text.items()}

#
# and now we combine document embeddings with TF-IDF vectors for each period's articles

X_data = {}
for period in articles_text.keys():
    embeds_vecs = np.concatenate((vectors[period], doc_embeds_tens[period]), axis=0)
    X_data[period] = torch.tensor(embeds_vecs, dtype=torch.float32)

#
# pad the tensors to the same length before creating the dataframe

feat_names = vectorizer.get_feature_names_out()
padding = pad_sequence(list(X_data.values()), batch_first=True)

#
# here we do some datetime manipulations to sort our index similarly in X and Y

datelist = list(articles_text.keys())
datelist.sort(key=lambda date: datetime.strptime(date, "%Y-%m-B%d"))

#
# create our X data

X = pd.DataFrame(padding.numpy(), index=datelist, columns=[period for period in range(padding.shape[1])])
X.columns = feat_names.tolist() + [f"col_{i}" for i in range(len(feat_names), len(X.columns))]
X.index.name = "Date"

#
# create our Y data and intersect our datasets

Y_data = feather.read_feather('E:/Tralgo/data/financial_container.csv')
Y = Y_data['TSLA_future_indicator']

tot_df = pd.merge(X, Y, how='inner', on='Date')
X = tot_df.iloc[:,0:-1]
Y = tot_df.iloc[:,-1]

#
# now we'll do some preprocessing of our data

mm_scaler = preprocessing.MinMaxScaler()
X_scale = mm_scaler.fit_transform(X)

X_train, X_val_test, Y_train, Y_val_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

#
# after this point we begin to code the neural network!