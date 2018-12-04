from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import Ridge
import pickle
from sklearn.linear_model import LogisticRegression
from data_cleaning import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
import os


def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


ratings = pd.read_csv('train_reviews.csv')
print("no of training data", ratings.shape)

businesses = pd.read_csv('business.csv')
print("no of bus", businesses.shape)

users = pd.read_csv('users.csv')
#users = None

#users = np.load('users.pkl')
print("no of users",users.shape)
#pickle.dump(users, open('users.pkl', 'wb'))


values = {'funny': 0,'useful':0,'review_count':0, 'average_stars':0}
users.fillna(value=values, inplace=True)

values = {'attributes_RestaurantsPriceRange2': 1,'attributes_BusinessAcceptsCreditCards':False,'stars':0, 'attributes_Alcohol':'none','attributes_RestaurantsAttire':'none',
          'attributes_WheelchairAccessible':False,'categories':'Restaurants'}
businesses.fillna(value=values, inplace=True)

cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BusinessAcceptsCreditCards')
cleaned_businesses,default_value_of_attributes_Alcohol = get_one_hot_encoded(businesses, 'attributes_Alcohol','none')
cleaned_businesses,default_value_of_RestaurantsAttire = get_one_hot_encoded(businesses, 'attributes_RestaurantsAttire','none')
cleaned_businesses,default_value_of_WheelchairAccessible = get_one_hot_encoded(businesses, 'attributes_WheelchairAccessible',False)
cleaned_businesses,default_value_of_categories = one_hot_encoded_multiclass(businesses, 'categories','Restaurants')



users = users[users['average_stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
u_stars = users['average_stars']
u_stars = u_stars.astype(float)
avg_user_rating= np.mean(u_stars)

businesses = businesses[businesses['stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
b_stars = businesses['stars']
b_stars = b_stars.astype(float)
mean_bussinesses_rating = np.mean(b_stars)

def scale_feautres(features):
    scaler = StandardScaler()
    feature = features.reshape(-1,1)
    scaled_feature = scaler.fit_transform(feature)
    features = scaled_feature.reshape(-1)
    return features

def extract_feautres(input_data,saved_feautres= None):
    feautres_of_train_data = np.zeros(shape=(input_data.shape[0], 2))

    if saved_feautres is not None and os.path.exists(saved_feautres):
        feautres_of_train_data = np.load(saved_feautres)
    else:
        for index, train in input_data.iterrows():
            features_extracted = []
            user_id = train.user_id
            b_id = train.business_id
            user_data = users[users.user_id == user_id]
            if(user_data.empty):
                funny_feautre = 0
                feautre_useful = 0
                review_count = 0
                feature_star = avg_user_rating
            else:
                funny_feautre = user_data.funny
                feautre_useful = user_data.useful
                review_count= user_data.review_count
                if(user_data.average_stars.values[0] == 0):
                    feature_star = avg_user_rating
                else:
                    feature_star = user_data.average_stars.values[0]
            b_data = cleaned_businesses[cleaned_businesses.business_id == b_id]
            if (b_data.empty):
                attributes_RestaurantsPriceRange2 = 1
                attributes_BusinessAcceptsCreditCards = 0
                feature_star_business = mean_bussinesses_rating
                attributes_Alcohol = default_value_of_attributes_Alcohol
                attributes_RestaurantsAttire = default_value_of_RestaurantsAttire
                attributes_WheelchairAccessible = default_value_of_WheelchairAccessible
                categories = default_value_of_categories

            else:
                attributes_RestaurantsPriceRange2 = b_data.attributes_RestaurantsPriceRange2
                attributes_BusinessAcceptsCreditCards = b_data.attributes_BusinessAcceptsCreditCards
                attributes_Alcohol = b_data.attributes_Alcohol
                attributes_RestaurantsAttire = b_data.attributes_RestaurantsAttire
                attributes_WheelchairAccessible = b_data.attributes_WheelchairAccessible
                categories = b_data.categories


                if (b_data.stars.values[0] == 0):
                    feature_star_business = mean_bussinesses_rating
                else:
                    feature_star_business = b_data.stars.values[0]

            features_extracted.append(feature_star)
            features_extracted.append(feature_star_business)

            '''
            features_extracted.append(funny_feautre)
            features_extracted.append(feautre_useful)
            features_extracted.append(review_count)
            features_extracted.append(attributes_RestaurantsPriceRange2)

            features_extracted.append(attributes_BusinessAcceptsCreditCards)
            features_extracted.append(attributes_Alcohol)
            features_extracted.append(attributes_RestaurantsAttire)
            features_extracted.append(attributes_WheelchairAccessible)
            features_extracted.append(categories)
            '''
            feautres_of_train_data[index] = np.array(features_extracted)

        pickle.dump(feautres_of_train_data, open(saved_feautres, 'wb'))
    for i in range(0, feautres_of_train_data.shape[1]):
        feautres_of_train_data[:, i] = scale_feautres(feautres_of_train_data[:, i])
    return feautres_of_train_data
#feautres_of_train_data = np.load('feautres_of_train_data.pkl')



feautres_of_train_data = extract_feautres( ratings, 'feautres_of_train_data4.pkl')

stars = ratings['stars']
print(stars.shape)

#clf = Ridge(alpha=1.0)
print("Fitting on train data..")
#clf = LinearSVC( C=0.001,solver='newton-cg', multi_class='multinomial',n_jobs=-1)
#clf = LinearSVC( C=0.001)
#clf = RandomForestClassifier()
#clf = BernoulliNB()
clf =DecisionTreeClassifier()
#clf = KNeighborsClassifier()
clf.fit(feautres_of_train_data, stars)

#test_data  = pd.read_csv('test_queries.csv', dtype={'user_id': str})
test_data  = pd.read_csv('test_with_gt.csv', dtype={'user_id': str})
test_data = test_data.drop(test_data[test_data.user_id == '#NAME?'].index)
test_data = test_data.drop(test_data[test_data.business_id == '#NAME?'].index)
test_data.to_csv('test_with_gt_cleaned.csv')

print(test_data.shape)
indices = []
print("Finding test data feautres..")
feautres_of_test_data = extract_feautres( test_data, 'feautres_of_test_data5.pkl')
feautres_of_test_data = feautres_of_test_data[0:test_data.shape[0]]
print("Predicting on test data feautres..")
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
        {
            'stars': prediction_values,
         })
test_submission_data.to_csv('out_validate_knn.csv')

