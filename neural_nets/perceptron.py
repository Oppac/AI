import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = np.random.choice([-1, 1], 2)
        self.learning_rate = 0.05

    def guess(self, input):
        sum = 0
        for i in range(len(input)):
            sum += input[i] * self.weights[i]
        return 1 if sum > 0 else -1

    def train(self, input, target):
        guess = self.guess(input)
        error = target - guess
        for i in range(len(self.weights)):
            self.weights[i] += error * input[i] * self.learning_rate

#####################################################

def line(x):
    return 0.3 * x - 1

def show(x, y, labels, p):
    colors = ['red', 'blue']
    plt.scatter(x=x, y=y, s=30, marker='o', c=labels, edgecolors='black',
                cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(x, line(x) , "k-")
    plt.show()

def main():
    size = 200
    width = 800; height = 800
    p = Perceptron()

    x = np.random.randint(0, width, size=size)
    y = np.random.randint(0, height, size=size)
    labels = [1 if y[i] > line(x[i]) else -1 for i in range(size)]
    inputs = [[x[i], y[i]] for i in range(size)]

    show(x, y, [p.guess(inputs[i]) for i in range(size)], p)
    while True:
        guesses = []
        for i in range(size):
            p.train(inputs[i], labels[i])
            guesses.append(p.guess(inputs[i]))
        show(x, y, guesses, p)
        if guesses == labels:
            break

main()
