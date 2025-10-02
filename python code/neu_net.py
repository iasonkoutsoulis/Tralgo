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
import re
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

try:
    torch.cuda.get_device_name(0)
except Exception:
    print('Warning: \nTorch not compiled with CUDA enabled.\nRunning on CPU ...')

#
# Fetch articles data

with open('/content/drive/MyDrive/Tralgo articles/article_container.json', 'r') as f:
    articles_text = json.load(f)

# #
# # this is something we only do for the local code to run

# import random
# random.seed(None)
# articles_text = dict(random.sample(list(articles_text.items()), 28))

# #
# # we tokenize our text data with their relevant tags from the dictionary

# tagged_arts = []
# for period, articles in articles_text.items():
#     for article in articles:
#         tag = period
#         tagged_arts.append(TaggedDocument(words=article.lower().split(), tags=[tag]))

# #
# # here we train our Doc2Vec model on every word of every article with attention to tags
# # note that vector_size dictates how many embeddings will pass into the X_data
# # on colab, this model is trained for 1000 epochs.

# # st = time.time()
# # print('Running doc2vec, maybe consider getting the google stuff')
# # doc2vec_model = Doc2Vec(tagged_arts, vector_size=128, min_count=10, epochs=10)
# # en = time.time()
# # print('Time elapsed: ', en-st)

# # doc2vec_model.save('/content/drive/MyDrive/Tralgo articles/d2v_M.model')

# #
# # next we'll work on the embeddings a bit
# # this d2v model is trained on Colab and downloaded here

# doc2vec_model = Doc2Vec.load('/content/drive/MyDrive/Tralgo articles/d2v_M.model')

# doc_embeds = {}
# for period, embeds in articles_text.items():
#     doc_embeds[period] = doc2vec_model.dv[period]

# doc_embeds_tens = {}
# for period, embeds in doc_embeds.items():
#     doc_embeds_tens[period] = np.array(embeds)

#
# in this part we transform words into vectors

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
vectors = {period: vectorizer.fit_transform(articles).toarray().astype(np.float32).flatten() for period, articles in articles_text.items()}

#
# and now we combine document embeddings with TF-IDF vectors for each period's articles

X_data = {}
for period in articles_text.keys():
    # embeds_vecs = np.concatenate((vectors[period], doc_embeds_tens[period]), axis=0)
    X_data[period] = torch.tensor(vectors[period], dtype=torch.float32)

#
# pad the tensors to the same length before creating the dataframe

padding = pad_sequence(list(X_data.values()), batch_first=True)

#
# here we create a date function to sort our index similarly in X and Y

def Datelist(input_dates):
    datelist = list(input_dates)
    datelist.sort(key=lambda date: datetime.strptime(date, "%Y-%m-B%d"))
    datelist = [re.sub(r'(?<=\d{4}-)\d{1}(?=-)', lambda match: match.group(0).zfill(2), date) for date in datelist]
    return datelist

#
# create our X data

X = pd.DataFrame(padding.numpy(), index=Datelist(articles_text.keys()), columns=[period for period in range(padding.shape[1])])
# feat_names = vectorizer.get_feature_names_out()
# X.columns = feat_names.tolist() + [f"col_{i}" for i in range(len(feat_names), len(X.columns))]
X.index.name = "Date"

#
# create our Y data and intersect our datasets

Y_data = feather.read_feather('/content/drive/MyDrive/Tralgo articles/financial_container.csv')
Y = pd.DataFrame(Y_data['GOOG_future_indicator'], index=Datelist(Y_data.index))
Y.index.name = "Date"

#
# we leave out the last row because Y is NaN and we'll use it for prediction.

tot_df = pd.merge(X, Y, how='inner', on='Date')
X = tot_df.iloc[0:-1,0:-1]
Y = tot_df.iloc[0:-1,-1]

#
# now we'll do some preprocessing of our data

mm_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X_scale = mm_scaler.fit_transform(X)

X_train, X_val_test, Y_train, Y_val_test = train_test_split(X_scale, Y, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

#
# after this point we begin to code the neural network!

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(len(X_train[0]), 256)
        # self.l2 = nn.Linear(64, 64)
        # self.l3 = nn.Linear(64, 64)
        # self.l4 = nn.Linear(64, 64)
        # self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(256, 128)
        self.l7 = nn.Linear(128, 64)
        self.l8 = nn.Linear(64, 2)
        self.l9 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.relu(self.l1(x))
        # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # x = self.relu(self.l2(x))
        # # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # x = self.relu(self.l3(x))
        # # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # x = self.relu(self.l4(x))
        # # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # x = self.relu(self.l5(x))
        # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.l6(x))
        # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.l7(x))
        # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.relu(self.l8(x))
        # x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.sigmoid(self.l9(x))
        return x

dropout_prob = 0.0
weight_decay = 0.01

model = NeuralNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

num_epochs = 300
batch_size = 32
best_val_loss = float('inf')
best_loss = float('inf')
patience = 10
for epoch in range(num_epochs):
    model.train()
    num_batches = len(X_train_tensor) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        inputs = X_train_tensor[start_idx:end_idx]
        targets = Y_train_tensor[start_idx:end_idx]

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Zero the gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Early stopping based on validation loss
    if (val_loss.item() < best_val_loss) & (loss.item() < best_loss):
        best_val_loss = val_loss.item()
        best_loss = loss.item()
        patience = 10  # Reset patience
    else:
        patience -= 1
        if patience == 0:
            print("Early stopping...")
            break

#
# if I ever want to load:
# remove the training part of the models
# ...
# model.load_state_dict(torch.load('/content/drive/MyDrive/Tralgo articles/model.pt'))

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_accuracy = ((test_outputs >= 0.0).float() == Y_test_tensor).float().mean()

print(f"Test Accuracy: {test_accuracy.item():.4f}")

torch.save(model.state_dict(), '/content/drive/MyDrive/Tralgo articles/model.pt')