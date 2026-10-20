import numpy as np
from three_layer_neural_network import NeuralNetwork, generate_data
from n_layer_neural_network import DeepNeuralNetwork


def numerical(f, P, h=1e-6):
    g = np.zeros_like(P)
    for i in np.ndindex(P.shape):
        old = P[i]
        P[i] = old + h
        fp = f()
        P[i] = old - h
        fm = f()
        P[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


X, y = generate_data()
X, y = X[:40], y[:40]
for act in ['tanh', 'sigmoid', 'relu']:
    m = NeuralNetwork(2, 5, 2, act, reg_lambda=0.0)
    m.feedforward(X, lambda x: m.actFun(x, act))
    dW1, dW2, db1, db2 = m.backprop(X, y)
    f = lambda: m.calculate_loss(X, y) * len(X)
    e3 = max(np.abs(g - numerical(f, P)).max() for g, P in [(dW1, m.W1), (dW2, m.W2), (db1, m.b1), (db2, m.b2)])
    d = DeepNeuralNetwork(2, 2, 5, 6, act, reg_lambda=0.0)
    d.feedforward(X, lambda x: d.actFun(x, act))
    grads = d.backprop(X, y)
    f = lambda: d.calculate_loss(X, y)
    e5 = max(max(np.abs(dW - numerical(f, L.W)).max(), np.abs(db - numerical(f, L.b)).max()) for L, (dW, db) in zip(d.layers, grads))
    print('%-8s 3-layer max abs error %.1e   5-layer max abs error %.1e' % (act, e3, e5))
