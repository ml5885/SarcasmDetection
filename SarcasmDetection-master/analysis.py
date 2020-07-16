import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import lime
import lime.lime_tabular
import lime.lime_text
import pickle
import re
import string

filepath = 'D:\mohsin\Sarcasm Model\Sarcastic Comments/train-balanced-sarcasm.csv'
df = pd.read_csv(filepath)
df = df.drop(columns=['author','subreddit','score','ups','downs','date','created_utc','parent_comment'])
df.dropna(inplace=True)

df['comment'] = df['comment'].map(lambda x: ''.join(e for e in x if e.isalnum() or e == ' '))
df['comment'] = df['comment'].map(lambda x: re.sub(r'\d+\w*|\d+', '', x))

# print(df['comment'].head())

train, test = train_test_split(df[:10000], random_state=42, test_size=0.20, shuffle=True)

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=10, ngram_range=(1,3))
vectorizer.fit(train['comment'])

x_train = vectorizer.transform(train['comment'])
y_train = train['label']
x_test = vectorizer.transform(test['comment'])
y_test = test['label']

print("training")
model = svm.SVC(probability=True)
model.fit(x_train, y_train)

print('scoring')
score = model.score(x_test, list(y_test))
print('Accuracy for {} data: {:.4f}'.format('sarcasm', score))

pickle.dump(model, open("models/models_100k.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer_100k.pkl", "wb"))

sent = "I don’t believe in plastic surgery. But in your case, go ahead"
vect_sent = vectorizer.transform([sent])
print(model.predict(vect_sent))
print(model.classes_)
print(model.predict_proba(vect_sent))

#classifier = svm.SVC()
#classifier.fit(x_train, y_train)
#score = classifier.score(x_test, y_test)


# sent = "Microsoft is giving up on physical retail. Today the company announced plans to permanently close all Microsoft Store locations in the United States and around the world, except for four locations that will be “reimagined” as experience centers that no longer sell products."
# vect_sent = vectorizer.transform([sent])
# print(classifier.predict(vect_sent))
# print(classifier.classes_)
# print(classifier.predict_proba(vect_sent))
#
# explainer = lime.lime_text.LimeTextExplainer(class_names=["business", "entertainment", "politics", "sport", "tech"])
#
# exp = explainer.explain_instance(sent, classifier_fn=pipeline.predict_proba, top_labels=1, num_features=10)
#
# output_file = '/Users/michaelli/PycharmProjects/textAnalysis/explanation.html'
# exp.save_to_file(output_file)

