import pandas as pd
import numpy as np
import sklearn.svm as svm
import pickle
from data_cleaning import *

train_reviews = pd.read_csv('train_reviews.csv')
train_reviews.fillna(0, inplace=True)
businesses = pd.read_csv('business.csv')
businesses.fillna(0, inplace=True)
users = pd.read_csv('users.csv')
users.fillna(0, inplace=True)

train_reviews_np = train_reviews.values
businesses_np = businesses.values


businessLe = idToNumber(businesses, 'business_id')
businessStarDict = dict(zip(businesses['business_id'], businesses['stars']))
userStarDict = dict(zip(users['user_id'], users['average_stars']))

reviewsTfidf = review_to_tfidf(train_reviews, 'text')


trainX = reviewsTfidf
trainX = np.zeros((reviewsTfidf.shape[0], reviewsTfidf.shape[1] + 3))
print(trainX.shape)

for i in range(len(reviewsTfidf)):
    trainX[i][reviewsTfidf.shape[1] + 0] = businessLe.transform([train_reviews_np[i][0]])
    trainX[i][reviewsTfidf.shape[1] + 1] = businessStarDict[train_reviews_np[i][0]]
    trainX[i][reviewsTfidf.shape[1] + 2] = userStarDict[train_reviews_np[i][8]]

print(trainX.shape)

trainY = train_reviews['stars']


lin_clf = svm.LinearSVC()
lin_clf.fit(trainX, trainY)

with open('lin_clf.joblib', 'wb') as f:
    pickle.dump(lin_clf, f)

print("fitted")

reviewZip = zip(train_reviews_np[0].tolist(), reviewsTfidf.tolist())
reviewDict = dict()
for (business, review) in reviewZip:
    if business in reviewDict:
        reviewDict[business] = (reviewDict[business][0] + review, reviewDict[business][1] + 1)
    else:
        reviewDict[business] = (review, 1)

for r in reviewDict:
    td = reviewDict[r][0]
    bot = reviewDict[r][1]
    reviewDict[r] = [ x / float(bot) for x in td ]

def predict(clf, business, user):

    input = []

    if business in reviewDict:
        input = reviewDict[business]
    else:
        input = [0] * 50

    print("Business: {}".format([business]))

    input.append(businessLe.transform([business]))
    input.append(businessStarDict[business])
    input.append(userStarDict[user])

    return clf.predict([input])


validate_queries = pd.read_csv('validate_queries.csv').values

print(validate_queries)

y_pred = []
for i in range(len(validate_queries)):
    print("validate_queries[i][2], business: {}".format(validate_queries[i][2]))
    print("validate_queries[i][1], user: {}".format(validate_queries[i][1]))

    result = predict(lin_clf, validate_queries[i][2], validate_queries[i][1])
    y_pred.append()


output = pd.DataFrame(
    {'stars': y_pred}
)

output.to_csv('linear_submission.csv')
