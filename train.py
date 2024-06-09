# in this file we will load the data and train our model

import torch
import torch.nn as nn

import numpy as np

import json
import textutils

from classes import ChatDataset, FeedForwardModel
from torch.utils.data import DataLoader

# read the data from the intents file

jsonData = None

with open("resources/intents.json", "r") as f:
    jsonData = json.load(f)

allWords = []
tags = []
xy = []

for intent in jsonData["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for patter in intent["patterns"]:
        words = textutils.tokenizeSentence(patter)

        allWords.extend(words)
        xy.append((words, tag))

# define some ignore/stop words
stopWords = ["?", "!", ".", ",", "-", "_", ";"]

# stem all words
allWords = [textutils.stem(w) for w in allWords if w not in stopWords]

# make all words and tags sorted sets, to omit duplicated
allWords = sorted(set(allWords))
tags = sorted(set(tags))

print(allWords[0: 20])
print(len(allWords))
print(tags)

X_train = []
y_train = []

# prepare the training data, the bag of words is our X, the labels our y
for (words, tag) in xy:
    bag = textutils.bagOfWords(words, allWords)
    label = tags.index(tag)

    X_train.append(bag)
    y_train.append(label)


# it's more efficient to create tensors from numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

# convert the data to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

# put the data in dataset and dataloader format

batchSize = 8

dataset = ChatDataset(X=X_train, y=y_train)
dataloader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True)

# training the model

hiddenLayerSize = 8
numberOfEpochs = 1200

model = FeedForwardModel(inputSize=len(allWords), hiddenSize=hiddenLayerSize, outputSize=len(tags))

# declare the loss function and optimizer
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(numberOfEpochs):

    for (words, labels) in dataloader:

        out = model(words)
        loss = lossFunction(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch} -> loss: {loss.item():.4f}")

# save the model and some metadata

data = {
    "model_state": model.state_dict(),
    "input_size": len(allWords),
    "hidden_size": 8,
    "output_size": len(tags),
    "all_words": allWords,
    "tags": tags
}

torch.save(data, "resources/chatmodel.pth")

print("### ok, all done")