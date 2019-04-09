# Dataset by MovieLens: https://grouplens.org/datasets/movielens/

import csv
import sys


def euclidian_similarity(user_ratings, user_1, user_2):
    print(list(user_ratings.values())[user_1])
    print()
    print(list(user_ratings.values())[user_2])

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
        user_1 = int(keys[int(sys.argv[1])-2])
        user_2 = int(keys[int(sys.argv[2])-2])
    else:
        user_1 = int(keys[0])
        user_2 = int(keys[1])
    euclidian_similarity(user_ratings,  user_1, user_2)


main()
