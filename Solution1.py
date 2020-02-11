from nltk.corpus import stopwords as sw
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import nltk
import re
import time


start_time = time.time()

#load the dataset
df = pd.read_csv("development.csv")
df_eval = pd.read_csv("evaluation.csv")
train_documents = df["text"]
eval_documents = df_eval["text"]
labels = np.array(df["class"])


#data exploration
cv = CountVectorizer(decode_error = 'ignore')
cv.fit(train_documents)
words = cv.get_feature_names()

#write words on file
with open('words.txt', 'w', encoding="utf-8") as fp:
    fp.write('words')
    for w in words:
        fp.write("%s\n" %(str(w)))

print("words written on words.txt")


#stemmer definition and analyzer function
it_stemmer = nltk.stem.SnowballStemmer('italian')

#this function is used to find words that contain numbers
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

#our analyzer stems words and substitutes number containing tokens with the "NNN"
#that is eventually removed by the inclusion of "NNN" in our stopwords.
def analyze_stem(word, stemmer):
    if not hasNumbers(word):
        return stemmer.stem(word)
    else:
        return "NNN"

#we build a class for a costum vectorizer that implements a stemmer as an analyzer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([analyze_stem(w, it_stemmer) for w in analyzer(doc)])

#TF-IDF extraction
my_sw = sw.words('italian')
my_sw.remove('non')
vectorizer_s = StemmedTfidfVectorizer(analyzer="word",
                                        stop_words=my_sw + ["nnn"], max_df = 0.3,
                                        min_df = 3, ngram_range = (1,2),
                                        max_features = 30000)
matrix = vectorizer_s.fit_transform(train_documents)
eval_matrix = vectorizer_s.transform(eval_documents)
print(matrix.shape)
print(eval_matrix.shape)
print("--- Vectorization time: %s seconds ---" % (time.time() - start_time))

#PCA via a trucated SVD
svd = TruncatedSVD(n_components = 100)
matrix_svd = svd.fit_transform(matrix)
eval_matrix_svd = svd.transform (eval_matrix)
print(matrix_svd.shape)
print(eval_matrix_svd.shape)

## This portion of code has been left commented to demonstrate the validation
## approach used
# #Hold-out testing
# X_train, X_test, y_train, y_test = train_test_split(matrix_svd, labels, test_size=0.20)
# mlp1 = MLPClassifier(hidden_layer_sizes = (125), alpha = 0.001, early_stopping = True, tol = 0.001, n_iter_no_change = 10, activation = 'identity')
# mlp1.fit(X_train, y_train)
# predictions = mlp1.predict(X_test)
# curr_score = f1_score(y_test, predictions, average='weighted')
# print(curr_score)

print("--- SVD time: %s seconds ---" % (time.time() - start_time))
# # Parameters tuning
# # The following code has been  commented since the computation of the best
# # parameters requires a very long time and has already been performed during the
# # testing procedure.
# # This function also implements cross validation so it is a nice way to test our
# # pipeline
# parameters = {'activation':['tanh', 'relu', 'logistic', 'identity'],
#               'alpha' : 10.0 ** -np.arange(0, 5),
#               'hidden_layer_sizes' : [(100),(125), (100,50), (100, 50, 25), (200,100,50)],
#               'early_stopping': [True], 'tol' :[0.001]}
# clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, cv = 7)
# clf.fit(matrix_svd, labels)
# print(clf.score(matrix_svd, labels))
# print(clf.best_params_)

#Training and prediction via our MLPClassifier
mlp1 = MLPClassifier(activation = 'identity',
                        alpha = 0.001,
                        hidden_layer_sizes = (125),
                        early_stopping = True, tol = 0.001,
                        n_iter_no_change = 10,
                        verbose = True)
mlp1.fit(matrix_svd,labels)
predictions = mlp1.predict(eval_matrix_svd)

#write on file
with open ('out1.csv', 'w') as f:
    i=0
    f.write('Id,Predicted\n')
    for p in predictions:
        f.write("%d,%s\n" %(i,p))
        i += 1
print("results written to out1.csv")
print("--- %s seconds ---" % (time.time() - start_time))
