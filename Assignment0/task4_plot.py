import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 600)

sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

d_sigmoid = sigmoid * (1 - sigmoid)
d_tanh = 1 - tanh**2
d_relu = (x > 0).astype(float)

names = ["Sigmoid", "Tanh", "ReLU"]
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
grid = "#e4e3df"
muted = "#52514e"

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.22)

panels = [
    (axes[0], [sigmoid, tanh, relu], r"Activation  $f(x)$", (-1.4, 3.2)),
    (axes[1], [d_sigmoid, d_tanh, d_relu], r"Derivative  $f'(x)$", (-0.1, 1.15)),
]

for ax, ys, title, ylim in panels:
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=muted, linewidth=0.8, alpha=0.6)
    ax.axvline(0, color=muted, linewidth=0.8, alpha=0.6)
    for y, colour, name in zip(ys, colors, names):
        ax.plot(x, y, color=colour, linewidth=2.0, label=name)
    ax.set_xlim(-6, 6)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x$", color=muted, fontsize=11)
    ax.tick_params(colors=muted, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)

axes[0].annotate("ReLU", xy=(2.6, 2.6), color=colors[2], fontsize=10, weight="bold")
axes[0].annotate("Tanh", xy=(3.2, 1.12), color=colors[1], fontsize=10, weight="bold")
axes[0].annotate("Sigmoid", xy=(1.9, 0.30), color=colors[0], fontsize=10, weight="bold")
axes[1].annotate("max 0.25", xy=(0, 0.25), xytext=(1.5, 0.45), color=muted, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=muted, linewidth=0.9))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           fontsize=10, labelcolor=muted, bbox_to_anchor=(0.5, -0.04))

fig.savefig("figures/task4.png", dpi=200, bbox_inches="tight")
