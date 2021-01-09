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
def create_utility_matrix  (users, movies, movie_rating):
    utility_matrix = np.zeros((len(movies) + 1, len(users) + 1))
    for row in movie_rating:
        utility_matrix[row[1], row[0]] = row[2]
    return utility_matrix


# normalizes matrix by subtracting row mean
def normalize_matrix(utility_matrix):
    normalized_matrix = np.copy(utility_matrix)
    for row in normalized_matrix:
        count = 0
        ratings_sum = 0
        for i in range(1, len(row)):                    # ignore column 0
            if row[i] != 0:
                count += 1
                ratings_sum += row[i]
        if count != 0:                                  # if movies doesnt have any ratings, there is nothing to normalize
            row_mean = ratings_sum / count
            for i in range(1, len(row)):
                if row[i] != 0:
                    row[i] = row[i] - row_mean          # subtract row mean from rating
    return normalized_matrix


# Calculates cosine similarity between two movies.
def movies_similarity(row1, row2):
    norm = np.linalg.norm(row1) * np.linalg.norm(row2)
    if norm == 0:                                       # one of the movies does not have any ratings
        return -1                                       # assume the movies are not similar at all
    return np.dot(row1, row2) / norm                    # cos α = A dot B / (|A| * |B|)


# Returns square matrix of similarities between every two movies
def movies_similarity_matrix(utility_matrix):
    number_of_movies = len(utility_matrix)
    similarities = np.full((number_of_movies, number_of_movies), -1)
    for i in range(1, len(utility_matrix)-1):           # ignore column 0
        for j in range(i+1, len(utility_matrix)):
            sim = movies_similarity(utility_matrix[i], utility_matrix[j])
            if sim > 1 or sim < -1:
                print("MOVIES SIMILARITY ERROR: SIMILARITY NOT WITHIN BOUNDS")
            similarities[i, j] = sim
            similarities[j, i] = sim
    return similarities


# find n most similar movies of the user similar to movie i
def get_n_similar(normalized_matrix, similarity_matrix, user_index, movie_index, n):
    row = similarity_matrix[movie_index]
    indexes = np.argsort(-1 * row)[:len(row)]           # vector of movies indexes sorted by greatest similarity first
    most_similar = np.empty((n, ))                      # TODO: not sure if its wise to use np.empty
    i = 0
    for index in indexes:
        if i == n - 1:                                  # only select n movies
            break
        if normalized_matrix[index, user_index] != 0:   # user has rated movie
            most_similar[i] = index
            i += 1
    return most_similar


###
#
# GLOBAL BASELINE UTILS
#
###

# Returns the average of all ratings.
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
    else:
        print("MEAN RATING ERROR: NO RATINGS PRESENT IN MATRIX")


# Returns bias of user's ratings.
def user_rating_deviation(utility_matrix, user_index, mean_rating):
    count = 0
    ratings_sum = 0

    user_column = utility_matrix[:, user_index]         # iterate over user's ratings
    for rating in user_column:
        if rating != 0:
            count += 1
            ratings_sum += rating
    if count == 0:
        return 0
    user_rating_avg = ratings_sum / count
    return user_rating_avg - mean_rating                # bias = user avg - global avg


# Returns bias of movie's ratings.
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
    return movie_rating_avg - mean_rating


# Returns array of rating deviations for all movies.
def movie_rating_deviation_matrix(utility_matrix, mean_rating):
    deviation_matrix = np.empty(len(utility_matrix))    # array of length = number of movies
    for index in range(1, len(utility_matrix)):
        deviation_matrix[index] = movie_rating_deviation(utility_matrix, index, mean_rating)
    return deviation_matrix


# Returns baseline estimate of user rating for movie, based on mean rating and biases.
def calculate_global_baseline(utility_matrix, user_index, movie_bias, mean_rating):
    user_deviation = user_rating_deviation(utility_matrix, user_index, mean_rating)
    return mean_rating + user_deviation + movie_bias    # Rxi = mean + Bx + Bi


#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CREATED UTILITY MATRIX")
    normalized_matrix = normalize_matrix(utility_matrix)
    print("NORMALIZED MATRIX")
    movies_similarities = movies_similarity_matrix(normalized_matrix)
    print("CREATED MOVIES SIMILARITIES")

    predictions_matrix = np.empty(len(predictions))
    predictions_np = predictions.to_numpy()

    n = 10
    i = 0
    for row in predictions_np:                          # matrix has user index on 1st column, movie index on 2nd
        user_index = row[0]
        movie_index = row[1]
        similar_ind = get_n_similar(normalized_matrix, movies_similarities, user_index, movie_index, n)

        similarity_rating_sum = 0
        similarity_sum = 0
        for item in similar_ind:
            if item is not 0:
                similarity_rating_sum += movies_similarities[movie_index, item] * utility_matrix[int(item), user_index]
                similarity_sum += movies_similarities[movie_index, int(item)]
        predictions_matrix[i] = similarity_rating_sum / similarity_sum
        i += 1
    return [[idx, predictions_matrix[idx-1]] for idx in range(1, len(predictions) + 1)]


# Combines Global Baseline with Item-Item collaborative Filtering
def predict_collaborative_filtering_V2(movies, users, ratings, predictions):
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CREATED UTILITY MATRIX")
    mean_rating = calculate_mean_rating(utility_matrix)
    print("CALCULATED MEAN RATING")
    normalized_matrix = normalize_matrix(utility_matrix)
    print("NORMALIZED MATRIX")
    movie_similarities = movies_similarity_matrix(normalized_matrix)
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


#####
##
## Global Baseline Estimate
##
#####

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

def calculate_gradient(matrix, step_size):
    gradient_matrix = np.zeros(matrix.shape)

    for row in range(0, len(matrix)):
        for value in range(0, len(matrix[row])):
            gradient_matrix[row][value] =


def predict_latent_factors(movies, users, ratings, predictions, k = 100):
    # R = Q * P
    # P and Q are mapped to k - dimensional rows
    # prediction for user x on movie i Rxi = Qi * PX
    # have P and Q such that it minimizes:
    # sum Of All Entrires not missing (rxi - qi * px) ^2
    # + lambda [sumx(length(px)^2) + sumi(length(qi)^2)]
    # lambda is a regularization parameter
    #
    # gradient descent: initialize P and Q (using SVD, missing ratings are 0)
    #                   Do gradient descent:
    #                   P <- P - n * derivative(P)
    #                   Q <- Q - n * derivative(Q)
    #                   n - learning rate
    # derivative(Q) = [derivative(qik)] and
    # derivative(qik) = sumxi(-2(rxi - qi*px)*pxk) + 2 * lambda * qik

    #movies on colums, users are rows
    utility_matrix = create_utility_matrix(users.to_numpy(), movies.to_numpy(), ratings.to_numpy())
    print("CALCULATING SVD")

    #might have to give it a transpose of utility
    U, S, VT = np.linalg.svd(utility_matrix)
    print(utility_matrix.shape, ' ', U.shape, ' ', S.shape, ' ', VT.shape, ' -- FINISHED')

    sigma = np.diag(S)
    #might not need to transpose
    V = VT.T

    print("APPROXIMATING TO P AND Q")
    Q = U[:, :k]
    P = sigma * VT
    P = Q[:, :k]


predict_latent_factors(movies_description, users_description, ratings_description, predictions_description)

#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):


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
#predictions = predict_collaborative_filtering_V2(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
# with open(submission_file, 'w') as submission_writer:
#     # Formates data
#     predictions = [map(str, row) for row in predictions]
#     predictions = [','.join(row) for row in predictions]
#     predictions = 'Id,Rating\n' + '\n'.join(predictions)
#
#     # Writes it dowmn
#     submission_writer.write(predictions)
