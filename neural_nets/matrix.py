from random import randint

class Matrix:
    def __init__(self, nb_rows, nb_cols, values=None):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        if values:
            self.values = values
        else:
            self.values = [[0 for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def randomize(self, lower, upper):
        self.values = [[randint(lower, upper)
                      for i in range(self.nb_cols)] for _ in range(self.nb_rows)]

    def add_scalar(self, scalar):
        self.values = [[j + scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]

    def multiply_scalar(self, scalar):
        self.values = [[j * scalar for j in self.values[i]]
                       for i in range(self.nb_rows)]

    def add_matrices(self, m2):
        m3 = Matrix(self.nb_rows, self.nb_cols)
        if (self.nb_rows == m2.nb_rows) and (self.nb_cols == m2.nb_cols):
            m3.values = [[j + k for j, k in zip(self.values[i], m2.values[i])]
                         for i in range(self.nb_rows)]
        else:
            print("Matrices must have the same size")
        return m3


    def multiply_matrices(self, m2):
        m3 = Matrix(self.nb_rows, m2.nb_cols)
        if (self.nb_cols == m2.nb_rows):
            for i in range(self.nb_rows):
                for j in range(m2.nb_cols):
                    for k in range(m2.nb_rows):
                        m3.values[i][j] += self.values[i][k] * m2.values[k][j]
        else:
            print("Invalid size: self.nb_cols != m2.nb_rows")
        return m3

    def transpose(self):
        trans = Matrix(self.nb_cols, self.nb_rows)
        for i in range(self.nb_rows):
            for j in range(self.nb_cols):
                trans.values[j][i] += self.values[i][j]
        return trans
