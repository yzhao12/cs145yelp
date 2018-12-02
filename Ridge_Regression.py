from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import Ridge
import pickle
from sklearn.linear_model import LogisticRegression
from data_cleaning import *

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


ratings = pd.read_csv('train_reviews.csv')
print("no of training data", ratings.shape)

businesses = pd.read_csv('business.csv')
print("no of bus", businesses.shape)

users = pd.read_csv('users.csv')
#users = np.load('users.pkl')
print("no of users",users.shape)
#pickle.dump(users, open('users.pkl', 'wb'))


values = {'funny': 0,'useful':0,'review_count':0}
users.fillna(value=values, inplace=True)

values = {'attributes_RestaurantsPriceRange2': 1,'attributes_BusinessAcceptsCreditCards':False}
businesses.fillna(value=values, inplace=True)

cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BusinessAcceptsCreditCards')
feautres_of_train_data = np.zeros(shape=(ratings.shape[0],5))

for index, train in ratings.iterrows():
    user_id = train.user_id
    b_id = train.business_id
    user_data = users[users.user_id == user_id]
    if(user_data.empty):
        funny_feautre = 0
        feautre_useful = 0
        feautre_review_count = 0
    else:
        funny_feautre = user_data.funny
        feautre_useful = user_data.useful
        review_count= user_data.review_count

    b_data = cleaned_businesses[cleaned_businesses.business_id == b_id]
    if (b_data.empty):
        attributes_RestaurantsPriceRange2 = 1
        attributes_BusinessAcceptsCreditCards = False
    else:
        attributes_RestaurantsPriceRange2 = b_data.attributes_RestaurantsPriceRange2
        attributes_BusinessAcceptsCreditCards = b_data.attributes_BusinessAcceptsCreditCards

    feautres_of_train_data[index,0] = funny_feautre
    feautres_of_train_data[index, 1] = feautre_useful
    feautres_of_train_data[index, 2] = review_count
    feautres_of_train_data[index, 3] = attributes_RestaurantsPriceRange2
    feautres_of_train_data[index, 4] = attributes_BusinessAcceptsCreditCards



#feautre_funny = ratings['funny']
#feautre_useful = ratings['useful']
stars = ratings['stars']
#final_features = np.vstack((feautre_funny,feautre_useful)).transpose()
#print(final_features.shape)
#print(feautre_funny.shape)
#print(feautre_useful.shape)
print(stars.shape)

#clf = Ridge(alpha=1.0)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
clf.fit(feautres_of_train_data, stars)

test_data  = pd.read_csv('test_queries.csv')
print(test_data.shape)
feautres_of_test_data = np.zeros(shape=(test_data.shape[0],5))
indices = []

for index, test in test_data.iterrows():
    user_id = test.user_id
    b_id = test.business_id
    user_data = users[users.user_id == user_id]
    if(user_data.empty):
        funny_feautre = 0
        feautre_useful = 0
        feautre_review_count = 0
    else:
        funny_feautre = user_data.funny
        feautre_useful = user_data.useful
        review_count= user_data.review_count

    b_data = cleaned_businesses[cleaned_businesses.business_id == b_id]
    if (b_data.empty):
        attributes_RestaurantsPriceRange2 = 1
        attributes_BusinessAcceptsCreditCards = False
    else:
        attributes_RestaurantsPriceRange2 = b_data.attributes_RestaurantsPriceRange2
        attributes_BusinessAcceptsCreditCards = b_data.attributes_BusinessAcceptsCreditCards

    feautres_of_test_data[index,0] = funny_feautre
    feautres_of_test_data[index, 1] = feautre_useful
    feautres_of_test_data[index, 2] = review_count
    feautres_of_test_data[index, 3] = attributes_RestaurantsPriceRange2
    feautres_of_test_data[index, 4] = attributes_BusinessAcceptsCreditCards
    indices.append(index)

#feautres_of_test_data = np.array(feautres_of_test_data)

prediction_values = clf.predict(feautres_of_test_data)
prediction_values_final = []
'''
for val in prediction_values:
    if(val <0):
        val_roundoff = 1
    if(val >0 )
    else:
        val_roundoff = float(round(prediction_values))
'''
test_submission_data = pd.DataFrame(
        {   'index': indices,
            'stars': prediction_values,
         })
test_submission_data.to_csv('out_ridge.csv')

