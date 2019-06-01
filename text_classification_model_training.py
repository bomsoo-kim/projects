import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def dataset_Amazon_Unlocked_Mobile():
    df = pd.read_csv('Amazon_Unlocked_Mobile.csv') # Read in the data

    df = df.sample(frac=0.1, random_state=10) # Sample the data to sp"eed up computation
    # df.head()

    df.dropna(inplace=True) # Drop missing values

    df = df[df['Rating'] != 3] # Remove any 'neutral' ratings equal to 3

    df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0) # Encode 4s and 5s as 1 (rated positively) and # Encode 1s and 2s as 0 (rated poorly)

    X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], df['Positively Rated'], random_state=0) # Split data into training and test sets
    return df, X_train, X_test, y_train, y_test

def dataset_spam():
    df = pd.read_csv('spam.csv')

    df['target'] = np.where(df['target']=='spam',1,0)
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], random_state=0)
    return df, X_train, X_test, y_train, y_test

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

#-------------------------------------------------------------------------------------
df, X_train, X_test, y_train, y_test = dataset_Amazon_Unlocked_Mobile()
# df, X_train, X_test, y_train, y_test = dataset_spam()

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train mean: ', y_train.mean())
print('y_test mean: ', y_test.mean())
print('X_train first entry: %s'%(X_train.iloc[0]))

df.head()

### feature engineering: the document-term matrix and some others #####################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
import re

vect = CountVectorizer().fit(X_train) # Fit the CountVectorizer to the training data
# vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train) # Fit the CountVectorizer to the training data specifiying a minimum document frequency of 5 and extracting 1-grams and 2-grams
# vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
# vect = TfidfVectorizer(min_df=5).fit(X_train) # Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
# vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)

X_train_vectorized = vect.transform(X_train) # transform the documents in the training data to a document-term matrix
X_test_vectorized = vect.transform(X_test) # transform the documents in the test data to a document-term matrix
print('X_train_vectorized shape: ', X_train_vectorized.shape)

# the number of characters
X_train_vectorized = add_feature(X_train_vectorized, X_train.map(lambda x: len(x))) # feature add 1
X_test_vectorized = add_feature(X_test_vectorized, X_test.map(lambda x: len(x))) # feature add 1

# the number of digits
X_train_vectorized = add_feature(X_train_vectorized, X_train.map(lambda x: len(re.findall('\d', x)))) # feature add 2
X_test_vectorized = add_feature(X_test_vectorized, X_test.map(lambda x: len(re.findall('\d', x)))) # feature add 2

# the number of non-words
X_train_vectorized = add_feature(X_train_vectorized, X_train.map(lambda x: len(re.findall('\W', x)))) # feature add 3
X_test_vectorized = add_feature(X_test_vectorized, X_test.map(lambda x: len(re.findall('\W', x)))) # feature add 3

# # get the feature names as numpy array
# feature_names = np.array(vect.get_feature_names())

# sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
# print('Smallest tfidf: {}'.format(feature_names[sorted_tfidf_index[:10]]))
# print('Largest tfidf: {}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

### Naive Bayes Classifier ##########################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

model = MultinomialNB(alpha = 0.1)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
print('AUC: ', roc_auc_score(y_test, y_pred))

# spam: 0.9720812182741116 : CountVectorizer().fit(X_train)
# spam: 0.9412063052815646 : TfidfVectorizer(min_df=5).fit(X_train)
# spam: 0.9437443763475543 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 1
# spam: 0.937055413136852 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 2
# spam: 0.9383095937388587 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 3
# spam: 0.9793070811333887 : CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
# spam: 0.9818451521993787 : CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train) : feature add 3

### Support Vector Machine ##########################################
from sklearn.svm import SVC # https://scikit-learn.org/stable/modules/svm.html
from sklearn.metrics import roc_auc_score

model = SVC(C = 10000)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
print('AUC: ', roc_auc_score(y_test, y_pred))

# spam: 0.9497160586048249 : TfidfVectorizer(min_df=5).fit(X_train)
# spam: 0.9581366823421557 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 1
# spam: 0.9598386330068077 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 2
# spam: 0.9657508955401253 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 2
# spam: 0.9759031798040846 : CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
# spam: 0.9792773712714121 : CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train) : feature add 3

### Logistic Regression ##########################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Train the model
# model = LogisticRegression()
model = LogisticRegression(C = 100)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized) # Predict the transformed test documents
print('AUC: ', roc_auc_score(y_test, y_pred))

# Sort the coefficients from the model # Find the 10 smallest and 10 largest coefficients
# sorted_coef_index = model.coef_[0].argsort()
# print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
# print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]])) # The 10 largest coefficients are being indexed using [:-11:-1] so the list returned is in order of largest to smallest

# spam: 0.9280978897509465 : CountVectorizer().fit(X_train)
# spam: 0.9577186221414868 : TfidfVectorizer(min_df=5).fit(X_train)
# spam: 0.9649147751387875 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 1
# spam: 0.9670347860041084 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 2
# spam: 0.9653328353394565 : TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train) : feature add 2
# spam: 0.9674528462047772 : TfidfVectorizer(min_df=5).fit(X_train) : feature add 3
# spam: 0.9742012291394326 : CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
# spam: 0.9788593110707434 : CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train) : feature add 3

#######################################################################
# # These reviews are treated the same by our current model
# print(model.predict(vect.transform(['not an issue, phone is working',
#                                     'an issue, phone is not working'])))
# print(model.predict(vect.transform(['this is not really good', 'this is not good'])))
