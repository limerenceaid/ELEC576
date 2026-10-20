# ELEC 576 / COMP 576 — Assignment 1

Report: [`ELEC576_Assignment1_Report.pdf`](ELEC576_Assignment1_Report.pdf)

## Layout

| Path | Part | What it is |
|---|---|---|
| `three_layer_neural_network.py` | 1 a–e | starter code filled in: activations, forward pass, loss, backprop |
| `n_layer_neural_network.py` | 1 f | `Layer` and `DeepNeuralNetwork(n_layers, layer_size)` |
| `check_gradients.py` | 1 d, f | finite-difference check of both backprop implementations |
| `run_part1.py` | 1 | all Part 1 experiments → `figures/p1*.png`, `outputs/part1_*.json` |
| `mnist_dcn.py` | 2 | DCN on MNIST with TensorBoard monitoring; `--act --init --opt --lr` |
| `run_part2.sh` | 2 | the 10 MNIST runs → `runs/<name>/` |
| `runs/` | 2 | TensorBoard event files and per-run `summary.json` |
| `tb_screenshots.py` | 2 | TensorBoard screenshots → `screenshots/` |
| `outputs/*.log` | 1, 2 | stdout of the NERSC runs |
| `ELEC576_Assignment1_Report.tex` | — | report source |

## Reproduce

On a Perlmutter login node:

```
module load pytorch/2.13.0
OMP_NUM_THREADS=4 python check_gradients.py
OMP_NUM_THREADS=4 python run_part1.py moons
OMP_NUM_THREADS=4 python run_part1.py digits
python run_part1.py digits_plot
bash run_part2.sh a
bash run_part2.sh b
```

Locally, with `tensorboard`, `playwright` and Google Chrome installed:

```
python tb_screenshots.py runs screenshots
tectonic ELEC576_Assignment1_Report.tex
```

`OMP_NUM_THREADS` matters: without it NumPy starts dozens of BLAS threads on the shared login node, and the small Part 1 matrices run far slower.
