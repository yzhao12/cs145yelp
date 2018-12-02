from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import Ridge
import pickle
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ratings = pd.read_csv('train_reviews.csv')
print("no of training data", ratings.shape)

businesses = pd.read_csv('business.csv')
print("no of bus", businesses.shape)

#users = pd.read_csv('users.csv')
#users = np.load('users.pkl')
#print("no of users",users.shape)
#pickle.dump(users, open('users.pkl', 'wb'))



def clean_boolean_feature(df,feature):
    feature_values = df[feature]
    cleaned_feauture_values = np.zeros(shape=(feature_values.shape))
    for idx, value in enumerate(feature_values):
        if value == True :
            cleaned_feauture_values[idx] =  1
        else:
            cleaned_feauture_values[idx] = 0
    df[feature] = cleaned_feauture_values
    return df


#clean_boolean_feature(businesses['attributes_BikeParking'],True)



values = {'attributes_BikeParking': False}
businesses.fillna(value=values, inplace=True)
b2 = clean_boolean_feature(businesses,'attributes_BikeParking')
print()