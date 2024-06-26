# https://docs.streamlit.io/get-started/fundamentals/main-concepts
import streamlit as st
import json
import torch
from classes import FeedForwardModel
import random
import textutils

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

botName = "Alice the bot"
DEBUG_MESSAGE_KEY = "debugMessage"

st.markdown("""
hello chatty, just talk with the bot or type :blue[!help] for some info.
""")


prompt = st.chat_input("Say something")
if prompt:
    # show available commands
    if prompt == "!help":
        st.info('''
        I am a very simple chat bot, just as me things.
        My topics mostly revolve around a dummy online shop.
        Available commands: 
        
        :blue[!categories] - see all the loaded/available categories
        
        :blue[!debug {on/off}] - set the debug messages to true / false
        ''')

    # show loaded/available categories/tags
    elif prompt == "!categories":
        st.info(f"Available categories: {tags}")

    elif prompt.startswith("!debug"):

        if prompt == "!debug on":
            st.session_state[DEBUG_MESSAGE_KEY] = "on"
            st.info(":blue[debug] set to on")

        if prompt == "!debug off":
            del st.session_state[DEBUG_MESSAGE_KEY]
            st.info(":blue[debug] set to off")

    # main business
    else:

        st.write(f"You: {prompt}")

        tokenized = textutils.tokenizeSentence(prompt)
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

        if prob > 0.7:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    st.write(f"{botName}: {random.choice(intent['responses'])}")
        else:
            st.write(f"{botName}: I can't understand, please rephrase")

        # this is a debug feature
        if DEBUG_MESSAGE_KEY in st.session_state:
            st.write(f":gray[[tag/category is {tag} with {prob} probability]]")