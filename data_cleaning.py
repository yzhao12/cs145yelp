from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import Ridge
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

'''
le = preprocessing.LabelEncoder()

ratings = pd.read_csv('train_reviews.csv')
print("no of training data", ratings.shape)

businesses = pd.read_csv('business.csv')
print("no of bus", businesses.shape)

#users = pd.read_csv('users.csv')
#users = np.load('users.pkl')
#print("no of users",users.shape)
#pickle.dump(users, open('users.pkl', 'wb'))

'''

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

def get_one_hot_encoded(df,feature,default_value_name):
    feature_values = df[feature]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(feature_values)
    default_value= label_encoder.transform([default_value_name])
    df[feature] = integer_encoded

    return df,default_value[0]

def get_one_hot_encoded_features(feature_values_unique,default_value_name):
    label_encoder = OneHotEncoder()
    integer_encoded = label_encoder.fit_transform(feature_values_unique)
    default_value= label_encoder.transform([default_value_name])

    return integer_encoded,default_value[0]

def one_hot_encoded_multiclass(df,feature,default_value_name):

    mlb = MultiLabelBinarizer()
    cat = df[feature].str.split(',')
    integer_encoded = mlb.fit_transform(cat)
    df[feature] = integer_encoded
    default_value = mlb.transform(np.array(default_value_name).reshape(-1, 1))
    #results_union = set().union(*cat)

    return df, default_value[0]

#clean_boolean_feature(businesses['attributes_BikeParking'],True)

'''

values = {'attributes_BikeParking': False}
businesses.fillna(value=values, inplace=True)
b2 = clean_boolean_feature(businesses,'attributes_BikeParking')
print()
'''