# ELEC 576 / COMP 576 — Assignment 0

Report: [`ELEC576_Assignment0_Report.pdf`](ELEC576_Assignment0_Report.pdf)

## Layout

| Path | Task | What it is |
|---|---|---|
| `outputs/task1_conda_info.txt` | 1 | `conda info` |
| `outputs/task1_conda_list.txt` | 1 | `conda list`, 552 packages |
| `task2_linalg.py` | 2 | drives all 82 rows of *Linear Algebra Equivalents* through IPython |
| `outputs/task2_transcript.txt` | 2 | the resulting transcript, one command per row |
| `task3_plot.py` → `figures/task3.png` | 3 | the assignment's plotting script |
| `task4_plot.py` → `figures/task4.png` | 4 | activation functions and their derivatives |
| `screenshots/task1_conda_info.png` | 1 | terminal capture pasted into the report |
| `make_report.py` | — | builds the PDF from the files above |

## Reproduce

```
conda create -n elec576 python=3.12 numpy scipy matplotlib ipython -y
conda activate elec576
pip install reportlab

ipython task2_linalg.py > outputs/task2_transcript.txt
ipython task3_plot.py
ipython task4_plot.py
python make_report.py
```

`task2_linalg.py` seeds with `default_rng(576)`, so the transcript is byte-identical across runs.

Environment used: macOS 15.1 arm64, Anaconda 2026.07-1, conda 26.5.3, Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, Matplotlib 3.11.0, IPython 9.15.0.
