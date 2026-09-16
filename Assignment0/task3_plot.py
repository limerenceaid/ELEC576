"""
ELEC 576 / COMP 576 - Assignment 0, Task 3
The script given in the assignment, run verbatim in IPython.
"""
import matplotlib
matplotlib.use("Agg")          # headless backend so the figure can be written to disk

# ---- the four lines exactly as printed in the assignment -------------------
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()
# ---------------------------------------------------------------------------

plt.savefig("figures/task3.png", dpi=200, bbox_inches="tight")
print("saved figures/task3.png")
