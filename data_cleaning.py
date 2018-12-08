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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

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

def review_to_tfidf(df, feature):
    reviewsText = df[feature]
    count_vect = CountVectorizer()
    reviewCounts = count_vect.fit_transform(reviewsText)

    tfidf_transformer = TfidfTransformer(use_idf=True)
    reviewTfidf = tfidf_transformer.fit_transform(reviewCounts)

    tsvd = TruncatedSVD(n_components=50)
    reduced_reviewTfidf = tsvd.fit_transform(reviewTfidf)

    return reduced_reviewTfidf

def idToNumber(df, feature):
    ids = df[feature]
    le = LabelEncoder()
    le.fit(ids)
    return le

def get_mulitple_features_from_one(df,feature,keys):
    feautre_values = df[feature].str.split(',').values
    final_feature_values = np.zeros(shape=(len(df), len(keys)))
    final_feature_values.fill(-1)
    #final_feature_values =[]
    for idx,value in enumerate(feautre_values):
        feature_value_for_single_data_point = np.zeros(shape=(len(keys)))
        feature_value_for_single_data_point.fill(-1)
        for idk, key_value_pair in enumerate(value) :
                val = str(key_value_pair).split()
                val = val[1]
                if(val == "False" or val == "False}"):
                    val = -1
                else:
                    val = 1
                feature_value_for_single_data_point[idk]= val

        #feature_value_for_single_data_point = np.array(feature_value_for_single_data_point)
        final_feature_values[idx]=  feature_value_for_single_data_point
    #final_feature_values = np.array(final_feature_values)
    for idx, key in enumerate(keys):
        df[key] = final_feature_values[:,idx]
    return df

#clean_boolean_feature(businesses['attributes_BikeParking'],True)

'''

values = {'attributes_BikeParking': False}
businesses.fillna(value=values, inplace=True)
b2 = clean_boolean_feature(businesses,'attributes_BikeParking')
print()
'''