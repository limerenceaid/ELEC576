__author__ = 'tan_nguyen'
import sys
import numpy as np
from sklearn import datasets, linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y, path=None, title=None):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(4, 3.2))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k', linewidths=0.3, s=18)
    if title:
        plt.title(title, fontsize=10)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200)
        plt.close()
    else:
        plt.show()

class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        type = type.lower()
        if type == 'tanh':
            return np.tanh(z)
        if type == 'sigmoid':
            return 1. / (1. + np.exp(-z))
        if type == 'relu':
            return np.maximum(z, 0)
        raise ValueError(type)

    def diff_actFun(self, z, type):
        type = type.lower()
        if type == 'tanh':
            return 1. - np.tanh(z) ** 2
        if type == 'sigmoid':
            s = 1. / (1. + np.exp(-z))
            return s * (1. - s)
        if type == 'relu':
            return (z > 0).astype(z.dtype)
        raise ValueError(type)

    def feedforward(self, X, actFun):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        exp_scores = np.exp(self.z2 - self.z2.max(axis=1, keepdims=True))
        self.probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        data_loss = -np.sum(np.log(self.probs[np.arange(num_examples), y]))
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        num_examples = len(X)
        delta2 = self.probs.copy()
        delta2[np.arange(num_examples), y] -= 1
        dW2 = self.a1.T.dot(delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = delta2.dot(self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type)
        dW1 = X.T.dot(delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            dW1, dW2, db1, db2 = self.backprop(X, y)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y, path=None, title=None):
        plot_decision_boundary(lambda x: self.predict(x), X, y, path, title)

def main():
    act = sys.argv[1] if len(sys.argv) > 1 else 'tanh'
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = sys.argv[3] if len(sys.argv) > 3 else None

    X, y = generate_data()
    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=hidden, nn_output_dim=2, actFun_type=act)
    model.fit_model(X, y)
    print("final loss %.4f  train acc %.3f" % (model.calculate_loss(X, y), np.mean(model.predict(X) == y)))
    model.visualize_decision_boundary(X, y, out, '%s, %d hidden units' % (act, hidden))

if __name__ == "__main__":
    main()
