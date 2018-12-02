import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

businesses = pd.read_csv('business.csv')
print("no of bus", businesses.shape)

#users = pd.read_csv('users.csv')
#print("no of users",users.shape)

ratings = pd.read_csv('train_reviews.csv')
print("no of training data", ratings.shape)


#print("no of unique users", users['user_id'].nunique())

n_bussinesses = ratings['business_id'].nunique()
print("no of unique bus", businesses['business_id'].nunique())

n_users = ratings['user_id'].nunique()
n_bussinesses = ratings['business_id'].nunique()
print("no of unique users", ratings['user_id'].nunique())
print("no of unique bus", ratings['business_id'].nunique())

train_data_matrix = np.zeros((n_users, n_bussinesses))

unique_users= ratings.user_id.unique()
unique_businesses= ratings.business_id.unique()

userid_dict = {}
i = 0
for user in unique_users:
    userid_dict[user] = i
    i = i+1

businessid_dict = {}
i = 0
for business in unique_businesses:
    businessid_dict[business] = i
    i = i+1

for index, rating_row in ratings.iterrows():
    user_id = rating_row.user_id
    b_id = rating_row.business_id
    user_id_index =  userid_dict[user_id]
    b_id_index = businessid_dict[b_id]
    train_data_matrix[user_id_index,b_id_index] = rating_row.stars

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
bussiness_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

test_data  = pd.read_csv('test_queries.csv')
print(test_data.shape)

# for index, rating_row in test_data.iterrows():
#     user_id = rating_row.user_id
#     b_id = rating_row.business_id
#     user_id_index =  userid_dict[user_id]
#     b_id_index = businessid_dict[b_id]
#     w_of_user = user_similarity[user_id_index]
#
#     mean_user_rating = mean_user_rating = ratings.mean(axis=1).mean(axis=1)
#     ratings_diff = (train_data_matrix - mean_user_rating[:, np.newaxis])
#     user_similarity.dot(ratings_diff)

def predict(ratings, similarity, type='user'):
    if type == 'user':
        #mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        #ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        #pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

        rating_trimmed = ratings[ratings > 0]
        mean_user_rating = rating_trimmed.mean(axis=1)
        # We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (rating_trimmed- mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T


        pred_values =  np.zeros((ratings.shape[0], ratings.shape[1]))
        for i in range(0,ratings.shape[0]):
            for j in range(0,ratings.shape[1]):
                users_who_rated_this_b = [k for k,v in enumerate(ratings[:,j] > 0) if v]
                list_ratings = np.array(ratings[:,j])
                ratings_of_users = list_ratings[users_who_rated_this_b]
                mean_rating = ratings_of_users.mean()
                sum = 0
                sum2 = 0
                for user in users_who_rated_this_b:
                    sum = sum + (list_ratings[user] - mean_rating) * similarity[i,user]
                    sum2  = sum2 + similarity[i,user]
                pred_values[i,j] = mean_rating +( sum/sum2)
        return pred_values

    elif type == 'item':
        pred_values = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    return pred_values

user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, bussiness_similarity, type='item')

prediction_values_for_user = []
prediction_values_for_b= []
indices = []
for index, rating_row in test_data.iterrows():
      user_id = rating_row.user_id
      b_id = rating_row.business_id
      user_id_index =  userid_dict[user_id]
      b_id_index = businessid_dict[b_id]

      prediction_user = float(round(user_prediction[user_id_index,b_id_index]))
      if(prediction_user == 0):
          prediction_user = 1
      prediction_values_for_user.append(prediction_user)

      prediction_b = float(round(item_prediction[user_id_index,b_id_index]))
      if(prediction_b== 0):
          prediction_b = 1


      prediction_values_for_b.append(prediction_b)
      indices.append(index)

test_submission_data = pd.DataFrame(
        {   'index': indices,
            'stars': prediction_values_for_user,
         })
test_submission_data.to_csv('out_colab_useruser.csv')


test_submission_data = pd.DataFrame(
        {   'index': indices,
            'stars': prediction_values_for_b,
         })
test_submission_data.to_csv('out_colab_bb.csv')