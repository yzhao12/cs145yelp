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
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LinearRegression
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

values = {'attributes_RestaurantsPriceRange2': 1,'attributes_BusinessAcceptsCreditCards':False,'stars':0, 'attributes_Alcohol':'none','attributes_RestaurantsAttire':'casual',
          'attributes_WheelchairAccessible':False,'categories':'Restaurants','attributes_WiFi':'no', 'attributes_BusinessParking':'{\'garage\': False, ''street\': False, \'validated\': False, \'lot\': False, \'valet\': False}',
          'attributes_GoodForMeal':'{\'dessert\': False, \'latenight\': False, \'lunch\': False, \'dinner\': True, \'breakfast\': False, \'brunch\': False}','attributes_RestaurantsTakeOut':False,
          'attributes_RestaurantsReservations':False,'attributes_DogsAllowed':False,'attributes_Ambience':'{\'romantic\': False, \'intimate\': False, \'classy\': False, \'hipster\': False, \'divey\' : False,  \'touristy\': False, \'trendy\': False, \'upscale\': False, \'casual\': False}',
          'attributes_BikeParking':False,'attributes_GoodForKids':False,'attributes_NoiseLevel':'average','attributes_OutdoorSeating':False,'attributes_RestaurantsGoodForGroups':False,
          'attributes_RestaurantsDelivery': False,'attributes_RestaurantsTableService':False,'attributes_WheelchairAccessible':False}
businesses.fillna(value=values, inplace=True)


cleaned_businesses = get_mulitple_features_from_one(businesses,'attributes_BusinessParking',['garage_BP','street_BP','validated_BP','Lot_BP','valet_BP' ])

cleaned_businesses = get_mulitple_features_from_one(businesses,'attributes_GoodForMeal',['dessert','latenight','lunch','dinner','breakfast','brunch'])
cleaned_businesses = get_mulitple_features_from_one(businesses,'attributes_Ambience',['romantic', 'intimate', 'classy', 'hipster','divey', 'touristy', 'trendy', 'upscale', 'casual'])
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BusinessAcceptsCreditCards')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BikeParking')

cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsTakeOut')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsReservations')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_DogsAllowed')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_GoodForKids')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_OutdoorSeating')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsGoodForGroups')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsDelivery')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsTableService')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_WheelchairAccessible')

cleaned_businesses,default_value_of_attributes_Alcohol = get_one_hot_encoded(businesses, 'attributes_Alcohol','none')
cleaned_businesses,default_value_of_RestaurantsAttire = get_one_hot_encoded(businesses, 'attributes_RestaurantsAttire','casual')
#cleaned_businesses,default_value_of_WheelchairAccessible = get_one_hot_encoded(businesses, 'attributes_WheelchairAccessible',False)
cleaned_businesses,default_value_of_attributes_WiFi = get_one_hot_encoded(businesses, 'attributes_WiFi','no')
cleaned_businesses,default_value_of_attributes_NoiseLevel = get_one_hot_encoded(businesses, 'attributes_NoiseLevel','average')

#cleaned_businesses,default_value_of_categories = one_hot_encoded_multiclass(businesses, 'categories','Restaurants')

'''
default_value_of_attributes_Alcohol = None
default_value_of_RestaurantsAttire = None
default_value_of_WheelchairAccessible = None
default_value_of_attributes_WiFi = None
default_value_of_attributes_NoiseLevel = None
default_value_of_categories = None
'''

users = users[users['average_stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
u_stars = users['average_stars']
u_stars = u_stars.astype(float)
avg_user_rating= np.mean(u_stars)

businesses = businesses[businesses['stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
b_stars = businesses['stars']
b_stars = b_stars.astype(float)
mean_bussinesses_rating = np.mean(b_stars)

def dump_value(featurename, value):
    pickle.dump(value,open(featurename+'.pkl', 'wb'))

def scale_feautres(features):
    scaler = StandardScaler()
    feature = features.reshape(-1,1)
    scaled_feature = scaler.fit_transform(feature)
    features = scaled_feature.reshape(-1)
    return features

def extract_feautres(input_data,train_or_test,saved_feautres= None):
    #feautres_of_train_data = np.zeros(shape=(input_data.shape[0], 2))

    if saved_feautres is not None and os.path.exists(saved_feautres):
        print("Found cached features..")
        feautres_of_train_data_new = []
        feautres_of_train_data = np.load(saved_feautres)
        feautres_of_train_data_new.append(feautres_of_train_data[:,0])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 1])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 4])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 6])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 5])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 10])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 25])
        feautres_of_train_data_new.append(feautres_of_train_data[:, 15])
        #feautres_of_train_data_new.append(feautres_of_train_data[:, 22])
        #feautres_of_train_data_new.append(feautres_of_train_data[:, 23])
        #feautres_of_train_data_new.append(feautres_of_train_data[:, 24])
        #feautres_of_train_data_new.append(feautres_of_train_data[:, 25])
        feautres_of_train_data_new = np.array(feautres_of_train_data_new).transpose()
        print("Returning cached features..")
    else:
        print("Extracting all features..")
        feautres_of_train_data = []
        print(input_data.shape)
        print(len(input_data.axes[0]))
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
                funny_feautre = user_data.funny.values[0]
                feautre_useful = user_data.useful.values[0]
                review_count= user_data.review_count.values[0]
                if(user_data.average_stars.values[0] == 0):
                    feature_star = avg_user_rating
                else:
                    feature_star = user_data.average_stars.values[0]
            b_data = cleaned_businesses[cleaned_businesses.business_id == b_id]
            if (b_data.empty):
                feature_star_business = mean_bussinesses_rating
                attributes_RestaurantsPriceRange2 = 1
                attributes_Alcohol = default_value_of_attributes_Alcohol
                attributes_BusinessAcceptsCreditCards = 0
                attributes_RestaurantsAttire = default_value_of_RestaurantsAttire
                attributes_WheelchairAccessible = False
                #categories = default_value_of_categories
                attributes_WiFi = default_value_of_attributes_WiFi
                attributes_RestaurantsTakeOut = False
                attributes_RestaurantsReservations = False
                attributes_DogsAllowed = False
                attributes_BikeParking = False
                attributes_GoodForKids = False
                attributes_NoiseLevel =default_value_of_attributes_NoiseLevel
                attributes_OutdoorSeating = False
                attributes_RestaurantsGoodForGroups = False
                attributes_RestaurantsDelivery = False
                attributes_RestaurantsTableService = False
                garage_BP = -1
                street_BP = -1
                validated_BP = -1
                Lot_BP = -1
                valet_BP = -1
                dessert  = -1
                latenight, =-1
                lunch =-1
                dinner = -1
                breakfast = -1
                brunch = -1
                romantic = -1
                intimate = -1
                classy = -1
                hipster = -1
                touristy = -1
                trendy = -1
                upscale =-1
                casual = 1

            else:
                '''
                features_extracted = b_data[['stars','attributes_RestaurantsPriceRange2','attributes_Alcohol','attributes_BusinessAcceptsCreditCards','attributes_RestaurantsAttire'
                    ,'attributes_WheelchairAccessible','attributes_WiFi','attributes_BusinessParking','attributes_GoodForMeal','attributes_RestaurantsTakeOut','attributes_RestaurantsReservations',
                            'attributes_DogsAllowed','attributes_Ambience','attributes_BikeParking','attributes_GoodForKids','attributes_NoiseLevel','attributes_OutdoorSeating',
                            'attributes_RestaurantsGoodForGroups','attributes_RestaurantsDelivery','attributes_RestaurantsTableService', 'garage_BP','street_BP','validated_BP','Lot_BP','valet_BP' ,
                            'dessert', 'latenight', 'lunch', 'dinner', 'breakfast', 'brunch',
                            'romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale',
                            'casual']]
                '''

                attributes_RestaurantsPriceRange2 = b_data.attributes_RestaurantsPriceRange2.values[0]
                attributes_BusinessAcceptsCreditCards = b_data.attributes_BusinessAcceptsCreditCards.values[0]
                attributes_Alcohol = b_data.attributes_Alcohol.values[0]
                attributes_RestaurantsAttire = b_data.attributes_RestaurantsAttire.values[0]
                attributes_WheelchairAccessible = b_data.attributes_WheelchairAccessible.values[0]
                #categories = b_data.categories.values[0]
                attributes_WiFi = b_data.attributes_WiFi.values[0]
                attributes_RestaurantsTakeOut = b_data.attributes_RestaurantsTakeOut.values[0]
                attributes_RestaurantsReservations = b_data.attributes_RestaurantsReservations.values[0]
                attributes_DogsAllowed = b_data.attributes_DogsAllowed.values[0]
                attributes_BikeParking = b_data.attributes_BikeParking.values[0]
                attributes_GoodForKids = b_data.attributes_GoodForKids.values[0]
                attributes_NoiseLevel = b_data.attributes_NoiseLevel.values[0]
                attributes_OutdoorSeating = b_data.attributes_OutdoorSeating.values[0]
                attributes_RestaurantsGoodForGroups = b_data.attributes_RestaurantsGoodForGroups.values[0]
                attributes_RestaurantsDelivery = b_data.attributes_RestaurantsDelivery.values[0]
                attributes_RestaurantsTableService = b_data.attributes_RestaurantsTableService.values[0]

                garage_BP = b_data.garage_BP.values[0]
                street_BP =b_data.street_BP.values[0]
                validated_BP = b_data.validated_BP.values[0]
                Lot_BP = b_data.Lot_BP.values[0]
                valet_BP =b_data.valet_BP.values[0]

                dessert = b_data.dessert.values[0]
                latenight = b_data.latenight.values[0]
                lunch = b_data.lunch.values[0]
                dinner = b_data.dinner.values[0]
                breakfast = b_data.breakfast.values[0]

                brunch = b_data.brunch.values[0]
                romantic = b_data.romantic.values[0]
                intimate = b_data.intimate.values[0]
                classy = b_data.classy.values[0]

                hipster = b_data.hipster.values[0]
                touristy = b_data.touristy.values[0]
                trendy = b_data.trendy.values[0]
                upscale = b_data.upscale.values[0]
                casual = b_data.casual.values[0]

                if (b_data.stars.values[0] == 0):
                    feature_star_business = mean_bussinesses_rating
                else:
                    feature_star_business = b_data.stars.values[0]

            features_extracted.append(feature_star) #0
            features_extracted.append(feature_star_business)#1

            features_extracted.append(funny_feautre)#2
            features_extracted.append(feautre_useful)#3
            features_extracted.append(review_count)#4
            features_extracted.append(attributes_RestaurantsPriceRange2)

            features_extracted.append(attributes_BusinessAcceptsCreditCards)#6
            features_extracted.append(attributes_Alcohol)
            features_extracted.append(attributes_RestaurantsAttire)
            features_extracted.append(attributes_WheelchairAccessible)
            #features_extracted.append(categories)
            features_extracted.append(attributes_WiFi)#10
            features_extracted.append(attributes_RestaurantsTakeOut)
            features_extracted.append(attributes_RestaurantsReservations)
            features_extracted.append(attributes_DogsAllowed)
            features_extracted.append(attributes_BikeParking)
            features_extracted.append(attributes_GoodForKids)#15
            features_extracted.append(attributes_NoiseLevel)
            features_extracted.append(attributes_OutdoorSeating)
            features_extracted.append(attributes_RestaurantsGoodForGroups)
            features_extracted.append(attributes_RestaurantsDelivery)
            features_extracted.append(attributes_RestaurantsTableService)#20
            features_extracted.append(garage_BP )
            features_extracted.append(street_BP )
            features_extracted.append(validated_BP )
            features_extracted.append(Lot_BP )
            features_extracted.append(valet_BP)#25
            features_extracted.append(dessert)
            features_extracted.append(latenight)
            features_extracted.append(lunch)
            features_extracted.append(dinner)
            features_extracted.append(breakfast)

            features_extracted.append(brunch)
            features_extracted.append(romantic)
            features_extracted.append(intimate)
            features_extracted.append(classy)

            features_extracted.append(hipster)
            features_extracted.append(touristy)
            features_extracted.append(trendy)
            features_extracted.append(upscale)
            features_extracted.append(casual)

            #dump_value('categories',categories)
            feautres_of_train_data.append(np.array(features_extracted))

        feautres_of_train_data = np.array(feautres_of_train_data)
        pickle.dump(feautres_of_train_data, open(saved_feautres, 'wb'))

        for i in range(0,feautres_of_train_data.shape[1]):
            dump_value('feature_' + str(i) + train_or_test, feautres_of_train_data[:, i])

        '''
        dump_value('feature_star' + train_or_test, feautres_of_train_data[:,0])
        dump_value('feature_star_business' + train_or_test, feautres_of_train_data[:,1])

        dump_value('funny'+ train_or_test, feautres_of_train_data[:,2])
        dump_value('feautre_useful'+ train_or_test, feautres_of_train_data[:,3])
        dump_value('review_count'+ train_or_test, feautres_of_train_data[:,4])
        dump_value('attributes_RestaurantsPriceRange2' + train_or_test, feautres_of_train_data[:,5])

        dump_value('attributes_BusinessAcceptsCreditCards' + train_or_test, feautres_of_train_data[:,6])
        dump_value('attributes_Alcohol' + train_or_test, feautres_of_train_data[:,7])

        dump_value('attributes_RestaurantsAttire' + train_or_test, feautres_of_train_data[:,8])
        dump_value('attributes_WheelchairAccessible'+train_or_test, feautres_of_train_data[:,9])
        #dump_value('categories' + train_or_test, feautres_of_train_data[:, 10])
        dump_value('attributes_WiFi'+train_or_test, feautres_of_train_data[:,11])
        #dump_value('attributes_BusinessParking' + train_or_test, feautres_of_train_data[:, 12])
        #dump_value('attributes_GoodForMeal' + train_or_test, feautres_of_train_data[:, 13])
        dump_value('attributes_RestaurantsTakeOut' + train_or_test, feautres_of_train_data[:, 14])
        dump_value('attributes_RestaurantsReservations' + train_or_test, feautres_of_train_data[:, 15])
        dump_value('attributes_DogsAllowed' + train_or_test, feautres_of_train_data[:, 16])
        #dump_value('attributes_Ambience' + train_or_test, feautres_of_train_data[:, 17])
        dump_value('attributes_BikeParking' + train_or_test, feautres_of_train_data[:, 18])
        dump_value('attributes_GoodForKids' + train_or_test, feautres_of_train_data[:, 19])
        dump_value('attributes_NoiseLevel' + train_or_test, feautres_of_train_data[:, 20])
        dump_value('attributes_OutdoorSeating' + train_or_test, feautres_of_train_data[:, 21])
        dump_value('attributes_RestaurantsGoodForGroups' + train_or_test, feautres_of_train_data[:, 22])
        dump_value('attributes_RestaurantsDelivery' + train_or_test, feautres_of_train_data[:, 23])
        dump_value('attributes_RestaurantsTableService' + train_or_test, feautres_of_train_data[:, 24])

        dump_value('garage_BP'+ train_or_test, feautres_of_train_data[:, 25])
        dump_value('street_BP' + train_or_test, feautres_of_train_data[:, 26])
        dump_value('validated_BP'+ train_or_test, feautres_of_train_data[:, 27])
        dump_value('Lot_BP'+ train_or_test, feautres_of_train_data[:, 28])
        dump_value('valet_BP'+ train_or_test, feautres_of_train_data[:, 29])

        dump_value('dessert'+ train_or_test, feautres_of_train_data[:, 30])
        dump_value('latenight'+ train_or_test, feautres_of_train_data[:, 31])
        dump_value('lunch'+ train_or_test, feautres_of_train_data[:, 32])
        dump_value('dinner'+ train_or_test, feautres_of_train_data[:, 33])
        dump_value('breakfast'+ train_or_test, feautres_of_train_data[:, 34])

        dump_value('brunch'+ train_or_test, feautres_of_train_data[:, 35])
        dump_value('romantic'+ train_or_test, feautres_of_train_data[:, 36])
        dump_value('intimate'+ train_or_test, feautres_of_train_data[:, 37])
        dump_value('classy'+ train_or_test, feautres_of_train_data[:, 38])

        dump_value('hipster'+ train_or_test, feautres_of_train_data[:, 40])
        dump_value('touristy'+ train_or_test, feautres_of_train_data[:, 41])
        dump_value('trendy'+ train_or_test, feautres_of_train_data[:, 42])
        dump_value('upscale'+ train_or_test, feautres_of_train_data[:, 43])
        dump_value('casual'+ train_or_test, feautres_of_train_data[:, 44])
        '''

    for i in range(0, feautres_of_train_data.shape[1]):
        feautres_of_train_data[:, i] = scale_feautres(feautres_of_train_data[:, i])
    for i in range(0, feautres_of_train_data_new.shape[1]):
        feautres_of_train_data_new[:, i] = scale_feautres(feautres_of_train_data_new[:, i])

    print("Returning features..")
    return feautres_of_train_data,feautres_of_train_data_new
#feautres_of_train_data = np.load('feautres_of_train_data.pkl')

feautres_of_train_data,feautres_of_train_data_new = extract_feautres( ratings,'train', 'feautres_of_train_data_set2.pkl')

stars = ratings['stars']
print(stars.shape)

#clf = Ridge(alpha=1.0)
print("Fitting on train data..")
#clf = LinearSVC( solver='newton-cg', multi_class='multinomial',n_jobs=-1)
#clf = LinearSVC( )
#clf = RandomForestClassifier()
#clf = BernoulliNB()
#clf =DecisionTreeClassifier()
#clf = KNeighborsClassifier()
clf = LinearRegression()
#clf = RidgeClassifier(solver='auto')
clf.fit(feautres_of_train_data_new,stars)
#clf.fit(feautres_of_train_data, stars)

#test_data  = pd.read_csv('test_queries.csv', dtype={'user_id': str})
test_data  = pd.read_csv('test_with_gt.csv', dtype={'user_id': str})
test_data = test_data.drop(test_data[test_data.user_id == '#NAME?'].index)
test_data = test_data.drop(test_data[test_data.business_id == '#NAME?'].index)
test_data.to_csv('test_with_gt_cleaned.csv')

print(test_data.shape)
indices = []
print("Finding test data feautres..")
#feautres_of_test_data,feautres_of_test_data_new = extract_feautres( test_data, 'test','feautres_of_test_data_newfeatures.pkl')
feautres_of_test_data,feautres_of_test_data_new = extract_feautres( test_data, 'test','feautres_of_validate_data_set2.pkl')
feautres_of_test_data = feautres_of_test_data[0:test_data.shape[0]]
print("Predicting on test data feautres..")
#prediction_values = clf.predict(feautres_of_test_data)
prediction_values = clf.predict(feautres_of_test_data_new)
prediction_values = np.round(list(prediction_values))
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
test_submission_data.to_csv('out_validate_1.csv')

