import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

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


# How to create np matrix from pd dataframe
# print(ratings_description.to_numpy())


# Returns utility matrix with users.size columns and movies.size rows.
# Values that are not present in the predictions array are set to zero.
# Params users, movies and movie_rating are numpy arrays.
# Row and column with index 0 should be ignored.
def create_utility_matrix(users, movies, movie_rating):
    utility_matrix = np.zeros((len(movies) + 1, len(users) + 1))
    for row in movie_rating:
        utility_matrix[row[1], row[0]] = row[2]
    return utility_matrix


###
# UTILS:
###
def calculate_mean_rating(utility_matrix):
    count = 0
    rating_sum = 0
    for row in utility_matrix:
        for rating in row:
            if rating != 0:
                count += 1
                rating_sum += rating
    if count != 0:
        return rating_sum / count


def user_rating_deviation(utility_matrix, user_index, mean_rating):
    count = 0
    ratings_sum = 0
    # iterate over user ratings
    user_column = utility_matrix[:, user_index]
    # print(user_column)
    for rating in user_column:
        if rating != 0:
            count += 1
            ratings_sum += rating
    if count == 0:
        return 0
    user_rating_avg = ratings_sum / count
    return user_rating_avg - mean_rating


def movie_rating_deviation(utility_matrix, movie_index, mean_rating):
    count = 0
    ratings_sum = 0
    for rating in utility_matrix[movie_index]:
        if rating != 0:
            count += 1
            ratings_sum += rating
    if count == 0:
        return 0
    movie_rating_avg = ratings_sum / count
    # print(movie_rating_avg)
    return movie_rating_avg - mean_rating


def movie_rating_deviation_matrix(utility_matrix, mean_rating):
    deviation_matrix = np.empty(len(utility_matrix))
    for index in range(1, len(utility_matrix)):
        deviation_matrix[index] = movie_rating_deviation(utility_matrix, index, mean_rating)

    return deviation_matrix


def calculate_global_baseline(utility_matrix, user_index, movie_deviation, mean_rating):
    user_deviation = user_rating_deviation(utility_matrix, user_index, mean_rating)
    # print("User Deviation", user_deviation)
    return mean_rating + user_deviation + movie_deviation


def normalize_matrix(utility_matrix):
    normalized_matrix = np.copy(utility_matrix)
    for row in normalized_matrix:
        count = 0
        ratings_sum = 0
        for i in range(0, len(row)):
            if row[i] != 0:
                count += 1
                ratings_sum += row[i]
        if count != 0:
            row_mean = ratings_sum / count
            for i in range(0, len(row)):
                if row[i] != 0:
                    row[i] = row[i] - row_mean
    return normalized_matrix


# Calculates cosine similarity between two movies.
def calculate_similarity(row1, row2):
    # cos α = A dot B / (|A| * |B|)
    norm = np.linalg.norm(row1) * np.linalg.norm(row2)
    if norm == 0:
        return -1
    return np.dot(row1, row2) / norm


# Returns array
def calculate_movies_similarity(utility_matrix):
    # len(utility_matrix) = number of movies
    similarities = np.empty((len(utility_matrix), len(utility_matrix)))
    for i in range(1, len(utility_matrix)-1):
        for j in range(i+1, len(utility_matrix)):
            sim = calculate_similarity(utility_matrix[i], utility_matrix[j])
            if sim > 1 or sim < -1:
                print("deep shit or dipshit")
            similarities[i, j] = sim
            similarities[j, i] = sim
    return similarities


# find most similar movies of the user similar to movie i
def get_n_similar(normalized_matrix, user, similarity_matrix, n, i):
    row = similarity_matrix[i]
    ind = np.argsort(-1 * row)[:len(row)]
    most_similar = np.empty((n, ))
    i = 0
    for index in ind:
        if i == n - 1:
            break
        if normalized_matrix[index, user] != 0:
            most_similar[i] = index
            i += 1
    return most_similar



# TESTING SUM SHIT
# matrix = np.array([[0, 0, 0, 0, 0],
#                   [0, 3, 0, 5, 4],
#                   [0, 4, 1, 5, 3],
#                   [0, 0, 5, 0, 2]])
# mean_rating = calculate_mean_rating(matrix)
# print("MEAN RATING", mean_rating)
# movie_deviation = movie_rating_deviation_matrix(matrix, mean_rating)
# for i in range(1, 5):
#     estimate = calculate_global_baseline(matrix, i, movie_deviation[3], mean_rating)
#     print("BASELINE ESTIMATE: ", estimate)
# # TESTING SIMILARITY:
# row1 = np.array([4, 0, 0, 5, 1, 0, 0])
# row2 = np.array([5, 5, 4, 0, 0, 0, 0])
# similarity = calculate_similarity(row1, row2)
# print(similarity)

# # TESTING:
# utility_matrix = create_utility_matrix(users_description.to_numpy(),
#                                        movies_description.to_numpy(),
#                                        ratings_description.to_numpy())
# mean_rating = calculate_mean_rating(utility_matrix)
# print("MEAN RATING: ", mean_rating)
# for idx in range(1, 10):
#     baseline_estimate = calculate_global_baseline(utility_matrix, idx, idx)
#     print("Baseline Estimate", baseline_estimate)


# print(utility_matrix)
# normalized_matrix = normalize_matrix(utility_matrix)
# print("Normalized matrix:")
# print(normalized_matrix)
# movies_similarities = calculate_movies_similarity(normalized_matrix)
# print("Similarities:")
# # I think the length of some rows is close to zero and division causes an error.
# print(movies_similarities)

#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TODO: refactor naming
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CREATED UTILITY MATRIX")
    normalized_matrix = normalize_matrix(utility_matrix)
    print("NORMALIZED MATRIX")
    movie_similarities = calculate_movies_similarity(normalized_matrix)
    print("CREATED MOVIES SIMILARITIES")

    predictions_matrix = np.empty(len(predictions))
    predictions_np = predictions.to_numpy()
    i = 0

    for row in predictions_np:
        similar_ind = get_n_similar(normalized_matrix, row[0], movie_similarities, 10, row[1])
        #predictions_matrix[i, 0] = int(i + 1)

        rating_sim = 0
        sim_sum = 0
        for item in similar_ind:
            if item is not 0:
                rating_sim += movie_similarities[row[1], int(item)] * utility_matrix[int(item), row[0]]
                sim_sum += movie_similarities[row[1], int(item)]
        predictions_matrix[i] = rating_sim / sim_sum
        i += 1
    #return predictions_matrix
    return [[idx, predictions_matrix[idx-1]] for idx in range(1, len(predictions) + 1)]


# Combines Global Baseline with Item-Item collaborative Filtering
def predict_collaborative_filtering_V2(movies, users, ratings, predictions):
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CREATED UTILITY MATRIX")
    mean_rating = calculate_mean_rating(utility_matrix)
    print("CALCULATED MEAN RATING")
    normalized_matrix = normalize_matrix(utility_matrix)
    print("NORMALIZED MATRIX")
    movie_similarities = calculate_movies_similarity(normalized_matrix)
    print("CREATED MOVIES SIMILARITIES")
    movie_deviation_matrix = movie_rating_deviation_matrix(utility_matrix, mean_rating)
    print("CREATED MOVIES DEVIATION MATRIX")

    predictions_matrix = np.empty(len(predictions))
    predictions_np = predictions.to_numpy()
    i = 0

    for row in predictions_np:
        print(i)
        user_deviation = mean_rating + user_rating_deviation(utility_matrix, row[0], mean_rating)
        movie_deviation = movie_deviation_matrix[row[1]]
        similar_ind = get_n_similar(normalized_matrix, row[0], movie_similarities, 10, row[1])
        #predictions_matrix[i, 0] = int(i + 1)

        rating_sim = 0
        sim_sum = 0
        for item in similar_ind:
            if item is not 0:
                item_baseline_estimate = user_deviation + movie_deviation_matrix[int(item)]
                rating_sim += movie_similarities[row[1], int(item)] * (utility_matrix[int(item), row[0]] - item_baseline_estimate)
                sim_sum += movie_similarities[row[1], int(item)]
        predictions_matrix[i] = user_deviation + movie_deviation + rating_sim / sim_sum
        i += 1
    #return predictions_matrix
    return [[idx, predictions_matrix[idx-1]] for idx in range(1, len(predictions) + 1)]


def predict_global_average_for_all(movies, users, ratings, predictions):
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CREATED UTILITY MATRIX")
    mean_rating = calculate_mean_rating(utility_matrix)
    print("CALCULATED MEAN RATING")
    return [[idx, mean_rating] for idx in range(1, len(predictions) + 1)]


def predict_baseline_estimate(movies, users, ratings, predictions):
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CREATED UTILITY MATRIX")
    mean_rating = calculate_mean_rating(utility_matrix)
    print("CALCULATED MEAN RATING")
    movie_deviation_matrix = movie_rating_deviation_matrix(utility_matrix, mean_rating)
    print("CREATED MOVIES DEVIATION MATRIX")

    predictions_matrix = np.empty(len(predictions))
    predictions_np = predictions.to_numpy()
    i = 0
    for row in predictions_np:
        print(i)
        predictions_matrix[i] = calculate_global_baseline(utility_matrix, row[0], movie_deviation_matrix[row[1]], mean_rating)
        i += 1

    return [[idx, predictions_matrix[idx-1]] for idx in range(1, len(predictions) + 1)]
#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_collaborative_filtering_V2(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
