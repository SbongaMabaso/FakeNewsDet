from nltk import stem
import numpy as np
import pandas as pd
import re   ##use for searching paragraph or text
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

## Data Pre-processing
news_data = pd.read_csv('/fake-news/train.csv')
print(news_data.shape)
print(news_data.head())

##Find the number of missing values in the dataset
news_data.isnull().sum()

##Dropping the missing values with empty string
news_data = news_data.fillna('')

##Merging the author names and news title
news_data['content'] = news_data['author']+' '+news_data['title']
print(news_data['content']) ##wil be used for predictions

##Separating the data & label
X = news_data.drop(columns='label', axis=1)
Y = news_data['label']

print(X)
print(Y)

##Stemming - is a process of reducing a word to its root word
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content) #get all alphabet
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# apply content column on stemming function
news_data['content'] = news_data['content'].apply(stemming)

print("Processed content:", news_data['content'])

##Separating data and label
X = news_data['content'].values
Y = news_data['label'].values

##Convering textual data to numerical data
vectorizer = TfidVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

##Splitting data to training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

## Training the model: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

##Evaluation
#accuracy score
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy Score: ", test_data_accuracy)


##Making Prediction
X_new = X_test[0]
prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print("Real news!")
else:
    print("Fake news!")