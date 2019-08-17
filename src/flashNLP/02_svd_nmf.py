import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn import decomposition
from scipy import linalg
import matplotlib.pyplot as plt

# Creating our dataset
categories = ["alt.atheism", "talk.religion.misc", "comp.graphics", "sci.space"]
remove = ("headers", "footers", "quotes")
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories, remove=remove)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories, remove=remove)
# print(newsgroups_train.filenames.shape, newsgroups_train.target.shape)
# print(newsgroups_train)

# Let's look at our data
# print("\n".join(newsgroups_train.data[:1]))
print(np.array(newsgroups_train.target_names)[newsgroups_train.target[:3]])

num_topics = 6
num_top_words = 8

##########################################
"""
## Stop-words, Stemming, Lemmatization:
1. Stemming and Lemmatization both generate the root form of the words.
2. Lemmatization uses rules about the language. The resulting tokens are actual words.
3. Stemming is poor-man's lemmatization.
4. Stemming is crude hueristic that chops off ends of the word.
5. The resulting tokens after stemming may not be actual words.
6. Stemming is faster.
"""

from sklearn.feature_extraction import stop_words
# print(sorted(list(stop_words.ENGLISH_STOP_WORDS))[:20])
import nltk
nltk.download("wordnet")

wnl = nltk.stem.WordNetLemmatizer()
porter = nltk.stem.PorterStemmer()

word_list = ["feet", "foot", "foots", "footing", "footer"]
# word_list = ["fly", "flies", "flying"]
# word_list = ["organize", "organizes", "organizing"]
# word_list = ["universe", "university", "universities"]

# using lemmatization
print(f"{[wnl.lemmatize(word) for word in word_list]}")

# using stemming
print(f"{[porter.stem(word) for word in word_list]}")

## Data Processing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
nltk.download("punkt")

vectorizer = CountVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(newsgroups_train.data).todense()
print(vectors.shape)
# print(vectors[-2:][-10:])
vocab = np.array(vectorizer.get_feature_names())
print(vocab.shape)
print(vocab[7000:7020])

#########################################################################################
## Singular Value Decomposition
#########################################################################################
