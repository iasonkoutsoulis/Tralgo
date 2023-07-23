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

doc2vec_model = Doc2Vec(tagged_arts, vector_size=300, min_count=10, epochs=1000)

#
# next we'll work on the embeddings a bit

doc_embeds = {}
for period, embeds in articles_text.items():
    doc_embeds[period] = doc2vec_model.dv[period]

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

padding = pad_sequence(list(X_data.values()), batch_first=True)

#
# here we do some datetime manipulations to sort our index similarly in X and Y

datelist = list(articles_text.keys())
datelist.sort(key=lambda date: datetime.strptime(date, "%Y-%m-B%d"))

#
# create our X data

X = pd.DataFrame(padding.numpy(), index=datelist, columns=[period for period in range(padding.shape[1])])
feat_names = vectorizer.get_feature_names_out()
X.columns = feat_names.tolist() + [f"col_{i}" for i in range(len(feat_names), len(X.columns))]
X.index.name = "Date"

#
# create our Y data and intersect our datasets

Y_data = feather.read_feather('E:/Tralgo/data/financial_container.csv')
Y = Y_data['AAPL_future_indicator']

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

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(len(X_train[0]), 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x
    
model = NeuralNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

num_epochs = 10000
batch_size = 1
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

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_accuracy = ((test_outputs >= 0.5).float() == Y_test_tensor).float().mean()

print(f"Test Accuracy: {test_accuracy.item():.4f}")

torch.save(model.state_dict(), 'E:/Tralgo/model.pt')

#model.load_state_dict(torch.load('E:/Tralgo/model.pt'))