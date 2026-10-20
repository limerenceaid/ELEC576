import json
import os
import sys
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork, generate_data
from n_layer_neural_network import DeepNeuralNetwork

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
parts = sys.argv[1:] or ['moons', 'digits', 'digits_plot']
acts = ['tanh', 'sigmoid', 'relu']
results = {}

X, y = generate_data()
Xt, yt = datasets.make_moons(2000, noise=0.20, random_state=1)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


def boundary(ax, model, title):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[plt.cm.Spectral(0.07), plt.cm.Spectral(0.93)])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k', linewidths=0.3, s=10)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def score(model):
    return {'train_acc': float(np.mean(model.predict(X) == y)),
            'test_acc': float(np.mean(model.predict(Xt) == yt))}


if 'moons' in parts:
    plt.figure(figsize=(4, 3.2))
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral, edgecolors='k', linewidths=0.3)
    plt.tight_layout()
    plt.savefig('figures/p1a_moons.png', dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, act in zip(axes, ['tanh', 'sigmoid', 'relu']):
        model = NeuralNetwork(2, 3, 2, actFun_type=act)
        model.fit_model(X, y, print_loss=False)
        r = score(model)
        r['loss'] = float(model.calculate_loss(X, y))
        results['e1_%s' % act] = r
        boundary(ax, model, '%s (train acc %.3f)' % (act, r['train_acc']))
        print('e1', act, r, flush=True)
    fig.tight_layout()
    fig.savefig('figures/p1e1_activations.png', dpi=200)
    plt.close(fig)

    hidden_dims = [1, 2, 3, 4, 5, 10, 20, 50]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for ax, h in zip(axes.ravel(), hidden_dims):
        model = NeuralNetwork(2, h, 2, actFun_type='tanh')
        model.fit_model(X, y, print_loss=False)
        r = score(model)
        r['loss'] = float(model.calculate_loss(X, y))
        results['e2_h%d' % h] = r
        boundary(ax, model, '%d hidden (train %.3f / test %.3f)' % (h, r['train_acc'], r['test_acc']))
        print('e2', h, r, flush=True)
    fig.tight_layout()
    fig.savefig('figures/p1e2_hidden.png', dpi=200)
    plt.close(fig)

    depths = [3, 4, 6, 10]
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.2))
    for i, act in enumerate(acts):
        for j, n in enumerate(depths):
            model = DeepNeuralNetwork(2, 2, n, 10, actFun_type=act)
            hist = model.fit_model(X, y, epsilon=0.1, num_passes=20000, print_loss=False)
            r = score(model)
            r['loss'] = float(hist[-1][1])
            results['f_moons_%s_L%d' % (act, n)] = r
            boundary(axes[i, j], model, '%s, %d layers (train %.3f / test %.3f)' % (act, n, r['train_acc'], r['test_acc']))
            print('f depth', act, n, r, flush=True)
    fig.tight_layout()
    fig.savefig('figures/p1f_depth.png', dpi=200)
    plt.close(fig)

    widths = [2, 3, 10, 50]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, w in zip(axes, widths):
        model = DeepNeuralNetwork(2, 2, 6, w, actFun_type='tanh')
        hist = model.fit_model(X, y, epsilon=0.1, num_passes=20000, print_loss=False)
        r = score(model)
        r['loss'] = float(hist[-1][1])
        results['f_moons_width%d' % w] = r
        boundary(ax, model, 'width %d (train %.3f / test %.3f)' % (w, r['train_acc'], r['test_acc']))
        print('f width', w, r, flush=True)
    fig.tight_layout()
    fig.savefig('figures/p1f_width.png', dpi=200)
    plt.close(fig)

if 'digits' in parts:
    digits = datasets.load_digits()
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0, stratify=digits.target)
    scaler = StandardScaler().fit(Xd_tr)
    Xd_tr, Xd_te = scaler.transform(Xd_tr), scaler.transform(Xd_te)
    for act in [a for a in acts if a in parts] or acts:
        out = {}
        for n in [3, 4, 6]:
            for w in [32, 128]:
                model = DeepNeuralNetwork(64, 10, n, w, actFun_type=act)
                hist = model.fit_model(Xd_tr, yd_tr, epsilon=0.1, num_passes=5000, print_loss=False, log_every=50)
                r = {'loss': float(hist[-1][1]),
                     'train_acc': float(np.mean(model.predict(Xd_tr) == yd_tr)),
                     'test_acc': float(np.mean(model.predict(Xd_te) == yd_te))}
                out['L%d_w%d' % (n, w)] = r
                if n == 6 and w == 32:
                    out['curve'] = hist
                print('f digits', act, n, w, r, flush=True)
        with open('outputs/part1_digits_%s.json' % act, 'w') as f:
            json.dump(out, f, indent=1)

if 'digits_plot' in parts:
    plt.figure(figsize=(4.5, 3.2))
    for act in acts:
        with open('outputs/part1_digits_%s.json' % act) as f:
            it, loss = zip(*json.load(f)['curve'])
        plt.plot(it, loss, label=act)
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/p1f_digits_loss.png', dpi=200)
    plt.close()

if 'moons' in parts:
    with open('outputs/part1_results.json', 'w') as f:
        json.dump(results, f, indent=1)
