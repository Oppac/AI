# Dataset by MovieLens: https://grouplens.org/datasets/movielens/

import csv
import math
import sys


def euclidian_distance(user_ratings, user_1, user_2):
    sum_squares = 0
    common = common_movies(user_ratings, user_1, user_2)
    if not common:
        return 0
    for i in common:
        diff = i[1] - i[2]
        sum_squares += diff * diff
    sqrt_diff = math.sqrt(sum_squares)
    similarity = len(common) / (1 + sqrt_diff)
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


def k_nearest_neighbors(user_ratings, k, user):
    similarity = []
    for i in user_ratings.keys():
        try:
            if not user == int(i):
                s = euclidian_distance(user_ratings, user, int(i))
                similarity.append([i, s])
        except:
            pass
    similarity.sort(key=lambda x:x[1], reverse=True)
    return similarity[:k]


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
    if len(sys.argv) == 2:
        user_1 = int(keys[int(sys.argv[1])-1])
    else:
        user_1 = int(keys[0])
    print(k_nearest_neighbors(user_ratings, 10, user_1))


main()
