import sys
import numpy as np
from three_layer_neural_network import NeuralNetwork, generate_data


class Layer(object):
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros((1, out_dim))

    def feedforward(self, x, actFun):
        self.x = x
        self.z = x.dot(self.W) + self.b
        self.a = actFun(self.z)
        return self.a

    def backprop(self, da, diff_actFun):
        dz = da * diff_actFun(self.z)
        self.dW = self.x.T.dot(dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
        return dz.dot(self.W.T)


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, nn_input_dim, nn_output_dim, n_layers, layer_size, actFun_type='tanh', reg_lambda=1e-3, seed=0):
        if np.isscalar(layer_size):
            hidden = [int(layer_size)] * (n_layers - 2)
        else:
            hidden = list(layer_size)
        assert len(hidden) == n_layers - 2
        self.sizes = [nn_input_dim] + hidden + [nn_output_dim]
        self.n_layers = n_layers
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        np.random.seed(seed)
        self.layers = [Layer(self.sizes[i], self.sizes[i + 1]) for i in range(n_layers - 1)]

    def feedforward(self, X, actFun):
        a = X
        for layer in self.layers[:-1]:
            a = layer.feedforward(a, actFun)
        z = self.layers[-1].feedforward(a, lambda z: z)
        exp_scores = np.exp(z - z.max(axis=1, keepdims=True))
        self.probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        data_loss = -np.mean(np.log(self.probs[np.arange(num_examples), y] + 1e-12))
        reg_loss = self.reg_lambda / 2 * sum(np.sum(np.square(layer.W)) for layer in self.layers)
        return data_loss + reg_loss

    def backprop(self, X, y):
        num_examples = len(X)
        delta = self.probs.copy()
        delta[np.arange(num_examples), y] -= 1
        delta /= num_examples
        da = self.layers[-1].backprop(delta, lambda z: 1.)
        for layer in reversed(self.layers[:-1]):
            da = layer.backprop(da, lambda z: self.diff_actFun(z, type=self.actFun_type))
        return [(layer.dW, layer.db) for layer in self.layers]

    def fit_model(self, X, y, epsilon=0.1, num_passes=20000, print_loss=True, log_every=1000):
        history = []
        for i in range(num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            grads = self.backprop(X, y)
            for layer, (dW, db) in zip(self.layers, grads):
                dW += self.reg_lambda * layer.W
                layer.W += -epsilon * dW
                layer.b += -epsilon * db
            if i % log_every == 0 or i == num_passes - 1:
                loss = self.calculate_loss(X, y)
                history.append((i, loss))
                if print_loss:
                    print("Loss after iteration %i: %f" % (i, loss))
        return history


def main():
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    layer_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    act = sys.argv[3] if len(sys.argv) > 3 else 'tanh'
    out = sys.argv[4] if len(sys.argv) > 4 else None

    X, y = generate_data()
    model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, n_layers=n_layers, layer_size=layer_size, actFun_type=act)
    model.fit_model(X, y)
    print("final loss %.4f  train acc %.3f" % (model.calculate_loss(X, y), np.mean(model.predict(X) == y)))
    model.visualize_decision_boundary(X, y, out, '%d layers x %d, %s' % (n_layers, layer_size, act))


if __name__ == "__main__":
    main()
