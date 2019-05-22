from math import exp
from random import uniform

class Matrix:
    def __init__(self, nb_rows=1, nb_cols=1, values=None):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.values = [[0 for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def give_values(self, values):
        self.nb_rows = len(values)
        try:
            self.nb_cols = len(values[0])
        except:
            self.nb_cols = 1
        self.values = values

    def randomize(self, lower, upper):
        self.values = [[uniform(lower, upper)
                      for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def add_scalar(self, scalar):
        result = Matrix(self.nb_rows, self.nb_cols)
        result.values = [[j + scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]
        return result

    def multiply_scalar(self, scalar):
        result = Matrix(self.nb_rows, self.nb_cols)
        result.values = [[j * scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]
        return result

    def add_vector(self, vector):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (vector.nb_cols == 1):
            m3.values = [[j + vector.values[i][0] for j in self.values[i]]
                         for i in range(self.nb_rows)]
        else:
            exit("{}x{} is not a vector".format(vector.nb_rows, vector.nb_cols))
        return m3

    def multiply_vector(self, vector):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (vector.nb_cols == 1):
            m3.values = [[j * vector.values[i][0] for j in self.values[i]]
                         for i in range(self.nb_rows)]
        else:
            exit("{}x{} is not a vector".format(vector.nb_rows, vector.nb_cols))
        return m3

    def add_matrices(self, m2):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (self.nb_rows == m2.nb_rows) and (self.nb_cols == m2.nb_cols):
            m3.values = [[j + k for j, k in zip(self.values[i], m2.values[i])]
                         for i in range(self.nb_rows)]
        else:
            exit("Invalid size addition: {}x{} != {}x{}".format(
                  self.nb_rows, self.nb_cols, m2.nb_rows, m2.nb_cols))
        return m3

    def multiply_matrices(self, m2):
        m3 = Matrix(self.nb_rows, m2.nb_cols)
        if (self.nb_cols == m2.nb_rows):
            for i in range(self.nb_rows):
                for j in range(m2.nb_cols):
                    for k in range(m2.nb_rows):
                        m3.values[i][j] += self.values[i][k] * m2.values[k][j]
        else:
            exit("Invalid size multiply: {} != {}".format(self.nb_cols, m2.nb_rows))
        return m3

    def transpose(self):
        trans = Matrix(self.nb_cols, self.nb_rows)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                trans.values[j][i] += self.values[i][j]
        return trans

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def apply_sigmoid(self):
        result = Matrix(self.nb_rows, self.nb_cols)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                result.values[i][j] += self.sigmoid(self.values[i][j])
        return result

    def apply_derivative_sigmoid(self):
        result = Matrix(self.nb_rows, self.nb_cols)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                result.values[i][j] += (self.sigmoid(self.values[i][j])
                                       * (1 - self.sigmoid(self.values[i][j])))
        return result

    def print_size(self):
        print("Size: {}x{}".format(self.nb_rows, self.nb_cols))
