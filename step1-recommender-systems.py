import numpy as np
import pandas as pd
import os.path
# -*- coding: utf-8 -*-

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


# Returns utility matrix with users.size columns and movies.size rows.
# Values that are not present in the predictions array are set to zero.
# Params users, movies and movie_rating are numpy arrays.
# Row and column with index 0 should be ignored.
def create_utility_matrix(users, movies, movie_rating):
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
        for i in range(1, len(row)):  # ignore column 0
            if row[i] != 0:
                count += 1
                ratings_sum += row[i]
        if count != 0:  # if movies doesnt have any ratings, there is nothing to normalize
            row_mean = ratings_sum / count
            for i in range(1, len(row)):
                if row[i] != 0:
                    row[i] = row[i] - row_mean  # subtract row mean from rating
    return normalized_matrix


# Calculates cosine similarity between two movies.
def movies_similarity(row1, row2):
    norm = np.linalg.norm(row1) * np.linalg.norm(row2)
    if norm == 0:  # one of the movies does not have any ratings
        return -1  # assume the movies are not similar at all
    return np.dot(row1, row2) / norm  # cos α = A dot B / (|A| * |B|)


# Returns square matrix of similarities between every two movies
def movies_similarity_matrix(utility_matrix):
    number_of_movies = len(utility_matrix)
    similarities = np.empty((number_of_movies, number_of_movies))
    # similarities = np.full((number_of_movies, number_of_movies), -1)
    for i in range(1, number_of_movies - 1):  # ignore column 0
        for j in range(i + 1, number_of_movies):
            sim = movies_similarity(utility_matrix[i], utility_matrix[j])
            if sim > 1 or sim < -1:
                print("MOVIES SIMILARITY ERROR: SIMILARITY NOT WITHIN BOUNDS")
            similarities[i, j] = sim
            similarities[j, i] = sim
    return similarities


# find n most similar movies of the user similar to movie i
def get_n_similar(normalized_matrix, similarity_matrix, user_index, movie_index, n):
    row = similarity_matrix[movie_index]
    indexes = np.argsort(-1 * row)[:len(row)]  # vector of movies indexes sorted by greatest similarity first
    most_similar = np.empty((n,))  # TODO: not sure if its wise to use np.empty
    i = 0
    for index in indexes:
        if i == n - 1:  # only select n movies
            break
        if normalized_matrix[index, user_index] != 0:  # user has rated movie
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

    user_column = utility_matrix[:, user_index]  # iterate over user's ratings
    for rating in user_column:
        if rating != 0:
            count += 1
            ratings_sum += rating
    if count == 0:
        return 0
    user_rating_avg = ratings_sum / count
    return user_rating_avg - mean_rating  # bias = user avg - global avg


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
    deviation_matrix = np.empty(len(utility_matrix))  # array of length = number of movies
    deviation_matrix[0] = 0
    for index in range(1, len(utility_matrix)):
        deviation_matrix[index] = movie_rating_deviation(utility_matrix, index, mean_rating)
    return deviation_matrix


# Returns array of rating deviations for all users. Ignore rating at index 0
def users_movie_deviation_matrix(utility_matrix, mean_rating):
    num_users = utility_matrix.shape[1]
    deviation_matrix = np.empty(num_users)    # array of length = number of users
    deviation_matrix[0] = 0
    for index in range(1, num_users):
        deviation_matrix[index] = user_rating_deviation(utility_matrix, index, mean_rating)
    return deviation_matrix


# Returns baseline estimate of user rating for movie, based on mean rating and biases.
def calculate_global_baseline(utility_matrix, user_index, movie_bias, mean_rating):
    user_deviation = user_rating_deviation(utility_matrix, user_index, mean_rating)
    return mean_rating + user_deviation + movie_bias  # Rxi = mean + Bx + Bi


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
    for row in predictions_np:  # matrix has user index on 1st column, movie index on 2nd
        user_index = row[0]
        movie_index = row[1]
        similar_ind = get_n_similar(normalized_matrix, movies_similarities, user_index, movie_index, n)

        similarity_rating_sum = 0
        similarity_sum = 0
        for item in similar_ind:
            if item is not 0:
                similarity_rating_sum += movies_similarities[movie_index, int(item)] * utility_matrix[
                    int(item), user_index]
                similarity_sum += movies_similarities[movie_index, int(item)]
        predictions_matrix[i] = similarity_rating_sum / similarity_sum
        i += 1
    return [[idx, predictions_matrix[idx - 1]] for idx in range(1, len(predictions) + 1)]


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
    n = 10
    i = 0

    for row in predictions_np:
        print(i)
        user_index = row[0]
        movie_index = row[1]
        user_deviation = mean_rating + user_rating_deviation(utility_matrix, user_index, mean_rating)
        movie_deviation = movie_deviation_matrix[row[1]]
        similar_ind = get_n_similar(normalized_matrix, movie_similarities, user_index, movie_index, n)

        rating_sim = 0
        sim_sum = 0
        for item in similar_ind:
            if item is not 0:
                item_baseline_estimate = user_deviation + movie_deviation_matrix[int(item)]
                rating_sim += movie_similarities[movie_index, int(item)] * (
                        utility_matrix[int(item), user_index] - item_baseline_estimate)
                sim_sum += movie_similarities[movie_index, int(item)]
        predictions_matrix[i] = min(user_deviation + movie_deviation + rating_sim / sim_sum, 5.0)
        i += 1
    return [[idx, predictions_matrix[idx - 1]] for idx in range(1, len(predictions) + 1)]


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
        predictions_matrix[i] = calculate_global_baseline(utility_matrix, row[0], movie_deviation_matrix[row[1]],
                                                          mean_rating)
        i += 1
    return [[idx, predictions_matrix[idx - 1]] for idx in range(1, len(predictions) + 1)]


#####
##
## LATENT FACTORS
##
#####
def gradient_descent(mean_rating, user_bias_matrix, movie_bias_matrix, P, Q, ratings, epochs=50, alpha=0.005, l=0.1, k=30):
    print("Gradient Descent")
    epoch = 0

    while epoch < epochs:
        print("-epoch", epoch)
        for count, row in enumerate(ratings):
            user_index = row[0] - 1
            movie_index = row[1] - 1

            #global value - mean rating of all user ratings + user bias rating + movie bias rating
            global_value = mean_rating + user_bias_matrix[user_index] + movie_bias_matrix[movie_index]
            global_baseline = np.full((k, ), global_value)

            # regularization - 2 * lambda * Pik/Qxk
            regularization = np.full((k, ), 2 * l)
            regularizationQ = np.multiply(regularization, Q[user_index])
            regularizationP = np.multiply(regularization, P[movie_index])

            #prediction - dot product (Q user row, P movie row)
            rating = np.full((k, ), row[2])
            prediction = np.full((k, ), np.dot(Q[user_index], P[movie_index]))

            #prediction + global
            prediction_global = np.add(global_baseline, prediction)
            #real rating - (prediction + global)
            rating_error = np.subtract(rating, prediction_global)
            rating_error = np.subtract(rating, prediction)

            a = np.multiply(-2, P[movie_index])
            b = np.multiply(a, rating_error)
            Qdiff = np.add(b, regularizationQ)

            a = np.multiply(-2, Q[user_index])
            b = np.multiply(a, rating_error)
            Pdiff = np.add(b, regularizationP)

            alpha_array = np.full((k, ), alpha)
            Q[user_index] -= np.multiply(alpha_array, Qdiff)
            P[movie_index] -= np.multiply(alpha_array, Pdiff)

        epoch += 1
    print("Finished")
    return P, Q

def predict_latent_factors(movies, users, ratings, predictions, k=30):
    print("Setting up the Utility Matrix")
    users = users.to_numpy()
    movies = movies.to_numpy()
    ratings = ratings.to_numpy()

    utility_matrix = create_utility_matrix(users, movies, ratings)

    mean_rating = calculate_mean_rating(utility_matrix)
    user_bias_matrix = users_movie_deviation_matrix(utility_matrix, mean_rating)
    movie_bias_matrix = movie_rating_deviation_matrix(utility_matrix, mean_rating)
    matrix = utility_matrix[1:, 1:]
    print("Finished")

    # latent factors using SVD, performed pretty poorly
    # print("Performing SVD")
    # U, S, V = np.linalg.svd(matrix.T)
    # print(U.shape, " -- U shape | ", S.shape, " -- Sigma shape | ", V.shape, " -- V shape")
    #
    # print("Initializing P and Q")
    # Sigma = np.diag(S)
    # Q = U[:, :k]                                  # Users
    # P = np.matmul(Sigma, V)                       # Movies
    # P = P[:k, :].T
    # print(Q.shape, " -- Q shape", P.shape, " -- P shape")

    P = np.random.default_rng().uniform(-1, 1, (len(movies), k))
    Q = np.random.default_rng().uniform(-1, 1, (len(users), k))
    predictions_np = predictions.to_numpy()
    predict = np.empty(len(predictions))
    i = 0

    P, Q = gradient_descent(mean_rating, user_bias_matrix, movie_bias_matrix, P, Q, ratings)

    print("Predicting")
    for row in predictions_np:
        user_index = row[0] - 1
        movie_index = row[1] - 1
        predict[i] = np.dot(Q[user_index], P[movie_index]) + mean_rating + movie_bias_matrix[movie_index] + user_bias_matrix[user_index]
        i += 1
    print("Finished")

    return [[idx, predict[idx - 1]] for idx in range(1, len(predictions) + 1)]


#####
##
## FINAL PREDICTORS
##
#####
def average_rating(movies, users, ratings, predictions):
    pred_c_f = predict_collaborative_filtering_V2(movies, users, ratings, predictions)
    pred_latent_factors = predict_latent_factors(movies, users, ratings, predictions)
    return [[idx, (pred_c_f[idx - 1][1] + pred_latent_factors[idx-1][1])/2] for idx in range(1, len(predictions) + 1)]


def predict_final(movies, users, ratings, predictions):
    return average_rating(movies, users, ratings, predictions)


#####
##
## SAVE RESULTS
##
#####
predictions = predict_final(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it down
    submission_writer.write(predictions)
