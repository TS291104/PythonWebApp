import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scikitplot as skplt

stopwords = pd.read_table('stopwords.txt')['a'].tolist()

# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def stopwordFiller(txt):
    terms = []
    blob = TextBlob(txt.replace("?","").lower())
    for t in blob.pos_tags:
        if t[0] in stopwords:
            terms.append(t[0])
        else:
            terms.append("_%s_" % t[1])
    return(" ".join(terms))

def reverseStopwordFiller(txt):
    terms = []
    blob = TextBlob(txt.replace("?","").lower())
    for t in blob.pos_tags:
        if t[0] in stopwords:
            terms.append("_%s_" % t[1])
        else:
            terms.append(t[0])
    return(" ".join(terms))


# Import corpus of training data and format text
corpus_data = pd.read_csv("labeled_training_dataset.csv")
corpus_data['term'] = corpus_data.question.apply(stopwordFiller)
corpus_data = corpus_data[corpus_data.Intent != 'miss']
corpus_data['Intent'] = corpus_data['Intent'].str.lower()

# Train vectorizer model
model_vec = TfidfVectorizer(ngram_range=(1,2),min_df=2)
model_vec.fit_transform(corpus_data.term.unique())
Xt = model_vec.transform(corpus_data.term)
le = preprocessing.LabelEncoder()
yhat = le.fit_transform(corpus_data.Intent)

# Train model
X_train, X_test, y_train, y_test = train_test_split(Xt, yhat, test_size=0.5)
model = RandomForestClassifier(n_estimators=1000,max_depth=10)
model.fit(X_train,y_train)

# Store trained model
# Note... future iteration, code with timestamp and version number. Directory should pull from most recent.
# We could theoretically store the binary dump of these in a DB or graph.
dump(model,"rf_intent_model.joblib")
dump(model_vec,"intent_model_vectorizer.joblib")
dump(le,"intent_label_encoder.joblib")
print("Model trained successfully")




