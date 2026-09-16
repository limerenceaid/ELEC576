"""
ELEC 576 / COMP 576 - Assignment 0, Task 4
A figure of my own: three activation functions and their derivatives.

The right-hand panel is the point of the figure - sigmoid's gradient peaks at
0.25 and decays to zero in both tails, which is the vanishing-gradient problem
that motivates ReLU in deep networks.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
MUTED   = "#52514e"
GRID    = "#e4e3df"
SERIES  = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, aqua

x = np.linspace(-6, 6, 600)

sigmoid  = 1 / (1 + np.exp(-x))
tanh     = np.tanh(x)
relu     = np.maximum(0, x)

d_sigmoid = sigmoid * (1 - sigmoid)
d_tanh    = 1 - tanh**2
d_relu    = (x > 0).astype(float)

names  = ["Sigmoid", "Tanh", "ReLU"]
funcs  = [sigmoid, tanh, relu]
derivs = [d_sigmoid, d_tanh, d_relu]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
fig.subplots_adjust(wspace=0.22)

for ax, ys, title, ylim in [
    (axes[0], funcs,  "Activation  $f(x)$",         (-1.4, 3.2)),
    (axes[1], derivs, "Derivative  $f^{\\prime}(x)$", (-0.1, 1.15)),
]:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, alpha=0.6, zorder=1)
    for y, colour, name in zip(ys, SERIES, names):
        ax.plot(x, y, color=colour, linewidth=2.0, label=name, zorder=3)
    ax.set_ylim(*ylim)
    ax.set_xlim(-6, 6)
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    ax.set_xlabel("$x$", color=MUTED, fontsize=11)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

# direct labels, placed where the curves are furthest apart
axes[0].annotate("ReLU",    xy=(2.6, 2.6),  color=SERIES[2], fontsize=10, weight="bold")
axes[0].annotate("Tanh",    xy=(3.2, 1.12), color=SERIES[1], fontsize=10, weight="bold")
axes[0].annotate("Sigmoid", xy=(1.9, 0.30), color=SERIES[0], fontsize=10, weight="bold")

axes[1].annotate("max 0.25", xy=(0, 0.25), xytext=(1.5, 0.45),
                 color=MUTED, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           fontsize=10, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Why deep networks moved from sigmoid to ReLU",
             color=INK, fontsize=13.5, weight="bold", y=1.0)

fig.savefig("figures/task4.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("saved figures/task4.png")
