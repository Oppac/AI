# Dataset by MovieLens: https://grouplens.org/datasets/movielens/

import csv
import math
import sys


def euclidian_similarity(user_ratings, user_1, user_2):
    sum_squares = 0
    common = common_movies(user_ratings, user_1, user_2)
    print(common)
    if not common:
        return 0
    for i in common:
        diff = i[1] - i[2]
        sum_squares += diff * diff
    sqrt_diff = math.sqrt(sum_squares)
    similarity = 1 / (1 + sqrt_diff)
    return similarity


def common_movies(user_ratings, user_1, user_2):
    movies_1 = {}
    movies_2 = {}
    common = []
    for i in list(user_ratings.values())[user_1]:
        movies_1[i[0]] = i[1]
    for i in list(user_ratings.values())[user_2]:
        movies_2[i[0]] = i[1]
    for i in movies_1.keys():
        if i in movies_2.keys():
            common.append([i, movies_1.get(i), movies_2.get(i)])
    return common


def read_by_user():
    user_ratings = {}
    with open("movie_dataset/ratings.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        for row in reader:
            if row[0] in user_ratings:
                user_ratings[row[0]].append([float(row[1]), float(row[2])])
            else:
                user_ratings[row[0]] = [[float(row[1]), float(row[2])]]
    return user_ratings

def main():
    user_ratings = read_by_user()
    keys = list(user_ratings.keys())
    if len(sys.argv) == 3:
        user_1 = int(keys[int(sys.argv[1])-1])
        user_2 = int(keys[int(sys.argv[2])-1])
    else:
        user_1 = int(keys[0])
        user_2 = int(keys[1])
    print(euclidian_similarity(user_ratings, user_1, user_2))


main()
