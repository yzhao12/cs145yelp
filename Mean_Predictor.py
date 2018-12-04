import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

businesses = pd.read_csv('business.csv')
print("no of bus", businesses.shape)

users = pd.read_csv('users.csv')
print("no of users",users.shape)

ratings = pd.read_csv('train_reviews.csv')
print("no of training data", ratings.shape)

print("no of unique users", users['user_id'].nunique())
print("no of unique bus", businesses['business_id'].nunique())


print("no of unique users", ratings['user_id'].nunique())
print("no of unique bus", ratings['business_id'].nunique())

stars = ratings['stars']
rating_avg = np.mean(stars)
#print(rating_avg)

users = users[users['average_stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]

avg_user_rating= users[['average_stars','user_id']]


unique_businesses= businesses.business_id.unique()
print(unique_businesses.shape)

mean_bussinesses_rating = businesses[['stars','business_id']]
print(mean_bussinesses_rating.shape)
# for single_business in unique_businesses:
#     business_temp = businesses[businesses.business_id == single_business]
#     print(business_temp.shape)
#     business_temp_rating = business_temp.stars
#     mean_bussiness_rating = np.mean(business_temp_rating)
#     mean_bussinesses_rating.append(mean_bussiness_rating)



a = avg_user_rating['average_stars']
avg_user_rating['average_stars'] = a - rating_avg
b = mean_bussinesses_rating['stars'].values
mean_bussinesses_rating['stars'] = b - rating_avg

test_data  = pd.read_csv('test_with_gt.csv')
print(test_data.shape)

sample_data  = pd.read_csv('sample_submission.csv')

prediction_values = []
indices = []
print(test_data.shape)
for index, test in test_data.iterrows():
    user_id = test.user_id
    b_id = test.business_id
    user = avg_user_rating[avg_user_rating.user_id == user_id]
    if(user.empty):
        mean_u_star = 0
    else:
        mean_u_star = user.average_stars
        mean_u_star = mean_u_star.values[0]
    busin = mean_bussinesses_rating[mean_bussinesses_rating.business_id == b_id]
    if (busin.empty):
        mean_b_star = 0
    else:
        mean_b_star = busin.stars
        mean_b_star = mean_b_star.values[0]

    prediction = float(round(mean_u_star + mean_b_star + rating_avg))
    prediction_values.append(prediction)
    indices.append(index)


test_submission_data = pd.DataFrame(
        {   'index': indices,
            'stars': prediction_values,
         })
test_submission_data.to_csv('out_mean_validate.csv')

    #if((businesses.business_id ==  b_id).any() == False):
        #print("B Not present")
    #if((users.user_id == user_id).any() == False):
        #print("User Not present")

