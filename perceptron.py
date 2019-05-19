import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self):
        self.bias = 0
        self.weights = np.random.choice([1, -1], 2)
        self.learning_rate = 0.7

    def guess(self, input):
        sum = np.dot(input, self.weights) + self.bias
        return 1 if sum > 0 else -1

    def train(self, input, target, guesses):
        guess = self.guess(input)
        guesses.append(guess)
        error = target - guess
        for i in range(len(self.weights)):
            self.weights[i] += error * input[i] * self.learning_rate
        self.bias += error * self.learning_rate

    def guess_line(self, x):
        m = -(self.bias/self.weights[1]) - (self.weights[0]/self.weights[1])
        print(m)
        b = -self.bias
        return m * x + b

#####################################################

def line(x):
    return 0.8 * x

def show(x, y, values, p):
    colors = ['red', 'blue']
    plt.scatter(x=x, y=y, s=30, marker='o', c=values, edgecolors='black',
                cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(x, line(x) , "k-")
    plt.plot(x, p.guess_line(x), "g-")
    plt.show()

def main():
    size = 200
    width = 800; height = 800
    p = Perceptron()

    # Training data
    x = np.random.randint(0, width, size=size)
    y = np.random.randint(0, height, size=size)
    labels = [1 if y[i] > line(x[i]) else -1 for i in range(size)]
    inputs = [[x[i], y[i]] for i in range(size)]

    #show(x, y, labels, p)
    show(x, y, [p.guess(inputs[i]) for i in range(size)], p)
    while True:
        guesses = []
        for i in range(size):
            p.train(inputs[i], labels[i], guesses)
        show(x, y, guesses, p)
        if guesses == labels:
            break
    print(p.weights)
    show(x, y, guesses, p)


main()
