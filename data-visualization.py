import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import step1-recommender-systems as utils

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)


def create_utility_matrix(users, movies, movie_rating):
    utility_matrix = np.zeros((len(movies) + 1, len(users) + 1))
    for row in movie_rating:
        utility_matrix[row[1], row[0]] = row[2]
    return utility_matrix


matrix = create_utility_matrix(users_description.to_numpy(),
                               movies_description.to_numpy(),
                               ratings_description.to_numpy())
print("Matrix created")


def movies_rating_avg(utility_matrix, movies):
    avgs = np.zeros((len(movies), 2))
    rows = utility_matrix[1:, :]
    i = 0
    for row in rows:
        count = 0
        ratings_sum = 0
        for rating in row:
            if rating > 0.5:
                count += 1
                ratings_sum += rating
        if count != 0:
            avg = ratings_sum / count
            avgs[i, 0] = avg                # movie rating avg
        avgs[i, 1] = int(movies[i, 1])      # movie year
        i += 1
    return avgs


# movies_avg = movies_rating_avg(matrix, movies_description.to_numpy())
# print("Calculated movies avg")

# plt.title("Average movie rating per release year")
# plt.xlabel("Release year")
# plt.ylabel("Avg rating")
# plt.scatter(movies_avg[:, 1], movies_avg[:, 0], color = 'brown')
# plt.show()


def user_sex_rating(utility_matrix, users):
    avgs = np.empty((len(users), 2))
    cols = utility_matrix[1:, 1:]

    for i in range(0, len(users)):
        ratings_sum = 0
        count = 0
        for rating in cols[:, i]:
            if rating > 0.5:
                count += 1
                ratings_sum += rating
            if count is not 0:
                avgs[i, 0] = ratings_sum / count
            else:
                avgs[i, 0] = 0
            if users[i, 1] == "M":              # user sex
                avgs[i, 1] = 0                  # 0 is cock
            else:
                avgs[i, 1] = 1                  # 1 is pussy
    return avgs


# users_sex_avg = user_sex_rating(matrix, users_description.to_numpy())
# print("Average calculated")
# print(users_sex_avg)
# users_avg_per_sex = np.array([0.0, 0.0])
# #print(users_avg_per_sex)
# male_mask = users_sex_avg[:, 1] == 0
# female_mask = ~male_mask
# male_ratings = users_sex_avg[male_mask, 0]
# print(male_ratings)
# female_mask = users_sex_avg[female_mask, 0]
#
# users_avg_per_sex[0] = np.average(male_ratings)
# users_avg_per_sex[1] = np.average(female_mask)
#
# print(users_avg_per_sex)

# plt.title("Average movie rating per user sex")
# plt.xlabel("Sex")
# plt.ylabel("Avg rating")
# plt.scatter(users_avg_per_sex, ["M", "F"])
# plt.show()


def user_age_rating(utility_matrix, users):
    avgs = np.zeros((len(users), 2))
    cols = utility_matrix[1:, 1:]

    for i in range(0, len(users)):
        ratings_sum = 0
        count = 0
        for rating in cols[:, i]:
            if rating > 0.5:
                count += 1
                ratings_sum += rating
            if count is not 0:
                avgs[i, 0] = ratings_sum / count
            avgs[i, 1] = users[i, 2]                # user age
    return avgs


# user_age_avg = user_age_rating(matrix, users_description.to_numpy())
# print("Average calculated")
# print(user_age_avg)
#
# plt.title("Average rating per age")
# plt.xlabel("Age")
# plt.ylabel("Avg rating")
# plt.scatter(user_age_avg[:, 1], user_age_avg[:, 0])
# plt.show()

