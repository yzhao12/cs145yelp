import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

businesses = pd.read_csv('business.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('train_reviews.csv')

print(users.shape)
print(users.head())

#businesses.columns = ['address', 'attributes_AcceptsInsurance']
#print(list(businesses.columns))

print(users['name'].nunique())

n_users = ratings.user_id.unique().shape[0]
n_business= ratings.business_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_business))
for line in ratings.itertuples():
    a = line[1] - 1
    b= line[2] - 1
    c = line[3]
    data_matrix[line[1]-1, line[2]-1] = line[3]


#users.columns = ['userID', 'Location', 'Age']
#ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
#ratings.columns = ['userID', 'ISBN', 'bookRating']

bussiness_user_rating_matrix = pd.read_csv('train_reviews.csv')
print("unique:user_id",bussiness_user_rating_matrix['user_id'].nunique())
print("unique:b_id",bussiness_user_rating_matrix['business_id'].nunique())
print(bussiness_user_rating_matrix['user_id'].count())
#model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
#model_knn.fit(bussiness_user_rating_matrix)
