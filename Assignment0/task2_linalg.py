"""
ELEC 576 / COMP 576 - Assignment 0, Task 2
One command per row of the "Linear Algebra Equivalents" table in
"NumPy for MATLAB Users", all 82 rows, executed in IPython.

Run with:  ipython task2_linalg.py
"""
import re
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

shell = InteractiveShell.instance()
shell.colors = "NoColor"
_n = 0


def run(cmd):
    global _n
    _n += 1
    print(f"In [{_n}]: {cmd}")
    with capture_output() as cap:
        res = shell.run_cell(cmd, store_history=False)
    body = cap.stdout.rstrip("\n")
    if res.result is not None and not body:
        body = repr(res.result)
    body = re.sub(r"^Out\[\d+\]:", f"Out[{_n}]:", body, flags=re.M)
    if body:
        print(body)
    if res.error_in_exec is not None:
        print(f"{type(res.error_in_exec).__name__}: {res.error_in_exec}")


def row(num, matlab, cmd):
    hdr = f"--- Row {num} --- MATLAB: {matlab} "
    print("\n" + hdr + "-" * max(3, 78 - len(hdr)))
    run(cmd)


print("--- Setup " + "-" * 68)
for s in [
    "import numpy as np",
    "import scipy.linalg",
    "from numpy.random import default_rng",
    "from scipy import signal",
    "from scipy.sparse.linalg import eigs, cg",
    "linalg = scipy.linalg",
    "np.set_printoptions(precision=4, suppress=True, linewidth=100)",
    "seed = default_rng(576)",
    "a = np.round(seed.random((5, 5)), 2)",
    "b = np.round(seed.random((5, 5)), 2)",
    "c = np.round(seed.random((5, 5)), 2)",
    "d = np.round(seed.random((5, 5)), 2)",
    "v = np.round(seed.random(5), 2)",
    "x = a.copy()",
    "m, n, q = 2, 2, 2",
    "ai = seed.integers(0, 8, (5, 5))",
    "bi = seed.integers(0, 8, (5, 5))",
    "spd = a @ a.T + 5 * np.eye(5)",
    "sig = np.round(seed.random(12), 2)",
    "a", "b", "v",
]:
    run(s)

TABLE = [
    (1,  "ndims(a)",                  "np.ndim(a)"),
    (2,  "numel(a)",                  "np.size(a)"),
    (3,  "size(a)",                   "np.shape(a)"),
    (4,  "size(a,n)",                 "a.shape[n-1]"),
    (5,  "[ 1 2 3; 4 5 6 ]",          "np.array([[1., 2., 3.], [4., 5., 6.]])"),
    (6,  "[ a b; c d ]",              "np.block([[a, b], [c, d]])"),
    (7,  "a(end)",                    "a[-1]"),
    (8,  "a(2,5)",                    "a[1, 4]"),
    (9,  "a(2,:)",                    "a[1, :]"),
    (10, "a(1:5,:)",                  "a[0:5, :]"),
    (11, "a(end-4:end,:)",            "a[-5:]"),
    (12, "a(1:3,5:9)",                "a[0:3, 4:9]"),
    (13, "a([2,4,5],[1,3])",          "a[np.ix_([1, 3, 4], [0, 2])]"),
    (14, "a(3:2:21,:)",               "a[2:21:2, :]"),
    (15, "a(1:2:end,:)",              "a[::2, :]"),
    (16, "a(end:-1:1,:)",             "a[::-1, :]"),
    (17, "a([1:end 1],:)",            "a[np.r_[:len(a), 0]]"),
    (18, "a.'",                       "a.T"),
    (19, "a'",                        "a.conj().T"),
    (20, "a * b",                     "a @ b"),
    (21, "a .* b",                    "a * b"),
    (22, "a./b",                      "a/b"),
    (23, "a.^3",                      "a**3"),
    (24, "(a > 0.5)",                 "(a > 0.5)"),
    (25, "find(a > 0.5)",             "np.nonzero(a > 0.5)"),
    (26, "a(:,find(v > 0.5))",        "a[:, np.nonzero(v > 0.5)[0]]"),
    (27, "a(:,find(v>0.5))",          "a[:, v.T > 0.5]"),
    (28, "a(a<0.5)=0",                "a2 = a.copy(); a2[a2 < 0.5] = 0; a2"),
    (29, "a .* (a>0.5)",              "a * (a > 0.5)"),
    (30, "a(:) = 3",                  "a2 = a.copy(); a2[:] = 3; a2"),
    (31, "y=x",                       "y = x.copy(); y"),
    (32, "y=x(2,:)",                  "y = x[1, :].copy(); y"),
    (33, "y=x(:)",                    "y = x.flatten(); y"),
    (34, "1:10",                      "np.arange(1., 11.)"),
    (35, "0:9",                       "np.arange(10.)"),
    (36, "[1:10]'",                   "np.arange(1., 11.)[:, np.newaxis]"),
    (37, "zeros(3,4)",                "np.zeros((3, 4))"),
    (38, "zeros(3,4,5)",              "np.zeros((3, 4, 5))"),
    (39, "ones(3,4)",                 "np.ones((3, 4))"),
    (40, "eye(3)",                    "np.eye(3)"),
    (41, "diag(a)",                   "np.diag(a)"),
    (42, "diag(v,0)",                 "np.diag(v, 0)"),
    (43, "rng(42,'twister'); rand(3,4)", "rng = default_rng(42); rng.random((3, 4))"),
    (44, "linspace(1,3,4)",           "np.linspace(1, 3, 4)"),
    (45, "[x,y]=meshgrid(0:8,0:5)",   "np.mgrid[0:9., 0:6.]"),
    (46, "(eval on a grid)",          "np.ogrid[0:9., 0:6.]"),
    (47, "[x,y]=meshgrid([1,2,4],[2,4,5])", "np.meshgrid([1, 2, 4], [2, 4, 5])"),
    (48, "(eval on a grid)",          "np.ix_([1, 2, 4], [2, 4, 5])"),
    (49, "repmat(a, m, n)",           "np.tile(a, (m, n))"),
    (50, "[a b]",                     "np.hstack((a, b))"),
    (51, "[a; b]",                    "np.vstack((a, b))"),
    (52, "max(max(a))",               "a.max()"),
    (53, "max(a)",                    "a.max(0)"),
    (54, "max(a,[],2)",               "a.max(1)"),
    (55, "max(a,b)",                  "np.maximum(a, b)"),
    (56, "norm(v)",                   "np.linalg.norm(v)"),
    (57, "a & b",                     "np.logical_and(a, b)"),
    (58, "a | b",                     "np.logical_or(a, b)"),
    (59, "bitand(a,b)",               "ai & bi"),
    (60, "bitor(a,b)",                "ai | bi"),
    (61, "inv(a)",                    "linalg.inv(a)"),
    (62, "pinv(a)",                   "linalg.pinv(a)"),
    (63, "rank(a)",                   "np.linalg.matrix_rank(a)"),
    (64, "a\\b",                      "linalg.solve(a, b)"),
    (65, "b/a",                       "linalg.solve(a.T, b.T).T"),
    (66, "[U,S,V]=svd(a)",            "U, S, Vh = linalg.svd(a); V = Vh.T; (U, S, V)"),
    (67, "chol(a)",                   "linalg.cholesky(spd)"),
    (68, "[V,D]=eig(a)",              "D, V = linalg.eig(a); (D, V)"),
    (69, "[V,D]=eig(a,b)",            "D, V = linalg.eig(a, b); (D, V)"),
    (70, "[V,D]=eigs(a,3)",           "D, V = eigs(a, k=3); (D, V)"),
    (71, "[Q,R]=qr(a,0)",             "Q, R = linalg.qr(a); (Q, R)"),
    (72, "[L,U,P]=lu(a)",             "P, L, U = linalg.lu(a); (P, L, U)"),
    (73, "conjgrad",                  "cg(spd, v)"),
    (74, "fft(a)",                    "np.fft.fft(a)"),
    (75, "ifft(a)",                   "np.fft.ifft(a)"),
    (76, "sort(a)",                   "np.sort(a)"),
    (77, "sort(a, 2)",                "np.sort(a, axis=1)"),
    (78, "[b,I]=sortrows(a,1)",       "I = np.argsort(a[:, 0]); (I, a[I, :])"),
    (79, "x = Z\\y",                  "Z = a; y = v; linalg.lstsq(Z, y)"),
    (80, "decimate(x, q)",            "signal.resample(sig, int(np.ceil(len(sig)/q)))"),
    (81, "unique(a)",                 "np.unique(a)"),
    (82, "squeeze(a)",                "np.zeros((1, 3, 1, 4)).squeeze().shape"),
]

for num, matlab, cmd in TABLE:
    row(num, matlab, cmd)

print(f"\n--- All {len(TABLE)} rows executed, {_n} cells total " + "-" * 24)
