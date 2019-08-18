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
"""
1.  We would expect that the wordds that appear most frequently in one topic would appear
    less frequently in other topics; otherwise that word wouldn't be a good choice to
    separate out the two topics.
    Hence, the matrix representing the topics are orthogonal.
2.  The SVD algorithm factorizes a matrix into three different matrices:
    a. matrix U: orthogonal columns
    b. matrix E: diagonal matrix
    c. matrix V: orthogonal rows
3.  SVD is exact decomposition, since the matrices it generates are big enough to cover
    the original matrix.

Uses of SVD:
a.  Latent Semantic Analysis (LSA)
"""

U, s, Vh = linalg.svd(vectors,full_matrices=False)
print(U.shape, s.shape, Vh.shape)

# note: s is a vector
# tip: use np.diag to convert a vector into diagonal matrix and vice-versa
print(s[:4])
print(np.diag(s)[:4,:])
print(np.diag(np.diag(s))[:4])

# Confirm that U, s, Vh are decompositions of matrix "vectors"
# tip: check the product of U, s, Vh to equate to "vectors"
vectors_decomposed = U @ np.diag(s) @ Vh
# print(vectors_decomposed - vectors)
print(np.allclose(vectors_decomposed, vectors))

# Confirm that U, Vh are orthogonal
# tip: 1. Check if they are row-wise or col-wise orthogonal
# tip: 2. Check if their norm is 1
# tip: 3. U is col-wise orthonormal. Vh is row-wise

print(np.allclose(U.T @ U, np.diag(np.ones(U.T.shape[0]))))
print(np.allclose(Vh @ Vh.T, np.diag(np.ones(Vh.shape[0]))))

print(np.allclose(U.T @ U, np.eye(U.shape[1])))
print(np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0])))

# What can we say about the singular values s
plt.plot(s)
plt.show()

#########################################################################################

def show_topics(a: np.array):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [" ".join(t) for t in topic_words]

print(show_topics(Vh[:10]))

#########################################################################################


