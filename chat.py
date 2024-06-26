# chat file, here we load the mode and handle the chat loop

import torch
from classes import FeedForwardModel
import textutils
import json
import random

# load the intents file

intents = None

with open("resources/intents.json", "r") as f:
    intents = json.load(f)

# load the model

data = torch.load("resources/chatmodel.pth")

inputSize = data["input_size"]
hiddenSize = data["hidden_size"]
outputSize = data["output_size"]
modelState = data["model_state"]
allWords = data["all_words"]
tags = data["tags"]

model = FeedForwardModel(inputSize=inputSize, hiddenSize=hiddenSize, outputSize=outputSize)
model.load_state_dict(modelState)
model.eval()

botName = "Alice"

while True:
    sentence = input("You: ")

    if sentence == "quit":
        print("See you later!")
        break

    tokenized = textutils.tokenizeSentence(sentence)
    # this will return the representation of our bag of words, an array with the size of allWords, one hot encoded
    bag = textutils.bagOfWords(tokenized, allWords)

    # we need to reshape our array, to be in the form [[1, 2, 3 .... len(allWords)]]
    bag = bag.reshape(1, bag.shape[0])
    X = torch.from_numpy(bag)

    out = model(X)
    # from the out tensor (which is the size of our classes) we have the value representing the probabilities, we should pick the largest one
    _, predicted = torch.max(out, dim=1)
    prediction = predicted.item()

    # here we get the tag name, from the tags array we get the index of our prediction
    tag = tags[prediction]

    # we apply softmax to the the percentages of the predictions
    probs = torch.softmax(out, dim=1)
    # we get the actual percentage for our prediction
    prob = probs[0][prediction]

    print(f"[tag is {tag} with {prob} probability]")

    if prob > 0.7:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: I can't understand, please rephrase")

print("### ok, all done")