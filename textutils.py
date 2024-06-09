# this file will hold text manipulation utility methods

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

NLTK_LOCAL_FOLDER = "D:\\python\\temp\\nltk"

nltk.download('punkt', download_dir=NLTK_LOCAL_FOLDER)
nltk.download('words', download_dir=NLTK_LOCAL_FOLDER)
nltk.download('stopwords', download_dir=NLTK_LOCAL_FOLDER)

nltk.data.path.append(NLTK_LOCAL_FOLDER)

stemmer = PorterStemmer()


def tokenizeSentence(sentence: str) -> [str]:
    return nltk.word_tokenize(sentence)


def stem(word: str) -> str:
    return stemmer.stem(word.lower())


def bagOfWords(tokenizedSentence, allWords):
    stemed = [stem(w) for w in tokenizedSentence]

    bag = np.zeros(len(allWords), dtype=np.float32)

    for i, w in enumerate(allWords):
        if w in stemed:
            bag[i] = 1.0

    return bag

print(tokenizeSentence("this is just a simulation"))