"""Build the Assignment 0 report PDF.  Run with:  python make_report.py"""
import os, textwrap, datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                                Image, PageBreak, KeepTogether)

STUDENT, RICE_ID = "Shirou Jing", "sj186"
GH_USER  = "limerenceaid"
REPO_URL = "https://github.com/limerenceaid/ELEC576"
OUT      = "ELEC576_Assignment0_Report.pdf"

INK, MUTED, RULE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d8d7d2")
CODEBG = colors.HexColor("#f4f4f2")

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontName="Times-Bold",
                       fontSize=19, leading=23, textColor=INK, spaceAfter=2)
BYLINE = ParagraphStyle("BYLINE", parent=ss["BodyText"], fontName="Times-Roman",
                        fontSize=9.5, leading=12, alignment=1, textColor=MUTED, spaceAfter=14)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Times-Bold",
                    fontSize=13, leading=16, spaceBefore=13, spaceAfter=5, textColor=INK)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Times-Bold",
                    fontSize=10.5, leading=13, spaceBefore=9, spaceAfter=4, textColor=INK)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Times-Roman",
                      fontSize=10, leading=13.5, spaceAfter=6, textColor=INK)
NOTE = ParagraphStyle("NOTE", parent=BODY, fontSize=9, leading=11.6, spaceAfter=2.5, textColor=MUTED)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontName="Courier", fontSize=6.8,
                      leading=8.1, textColor=INK, backColor=CODEBG,
                      borderPadding=4, spaceBefore=2, spaceAfter=6)
CODE_BIG = ParagraphStyle("CODE_BIG", parent=CODE, fontSize=8, leading=9.8)


def code(text, style=CODE, width=None):
    if width is None:
        usable = letter[0] - 1.7 * inch - 2 * style.borderPadding
        width = int(usable / (style.fontSize * 0.6)) - 1
    out = []
    for line in text.rstrip("\n").split("\n"):
        if len(line) <= width:
            out.append(line)
        else:
            ind = " " * (len(line) - len(line.lstrip()))
            out += textwrap.wrap(line, width=width, subsequent_indent=ind + "    ",
                                 break_long_words=True, break_on_hyphens=False) or [line]
    return Preformatted("\n".join(out), style)


def shot(name, width=6.6):
    path = f"screenshots/{name}.png"
    if not os.path.exists(path):
        return None
    iw, ih = ImageReader(path).getSize()
    w = min(width, 6.6) * inch
    return Image(path, width=w, height=w * ih / iw)


def read(p):
    with open(p) as f:
        return f.read()


S = [Paragraph("Assignment 0", TITLE),
     Paragraph(f"{STUDENT} &nbsp;&middot;&nbsp; Rice ID: {RICE_ID}", BYLINE)]

# --------------------------------------------------------------- task 1 ----
S += [Paragraph("1&nbsp;&nbsp;Anaconda", H1),
      Paragraph(
          "<font face='Courier'>Anaconda3-2026.07-1-MacOSX-arm64.sh</font> installed into "
          "<font face='Courier'>~/anaconda3</font> on macOS 15.1 (arm64), followed by "
          "<font face='Courier'>conda init zsh</font>. Result: conda 26.5.3 on Python 3.14.6, "
          "552 packages in <font face='Courier'>base</font>. The full "
          "<font face='Courier'>conda list</font> is in the repository as "
          "<font face='Courier'>outputs/task1_conda_list.txt</font>.", BODY),
      Paragraph("Task 1 &mdash; <font face='Courier'>conda info</font>", H2)]
S += [shot("task1_conda_info") or code(read("outputs/task1_conda_info.txt"), CODE_BIG)]

# --------------------------------------------------------------- task 2 ----
S += [Paragraph("2&nbsp;&nbsp;MATLAB to Python &mdash; Linear Algebra Equivalents", H1),
      Paragraph(
          "All 82 rows of the table were run in IPython, one command per row. My matrices: "
          "<font face='Courier'>a, b, c, d</font> are 5&times;5 drawn uniformly from [0,1) and "
          "rounded to two decimals, <font face='Courier'>v</font> a matching 5-vector, seeded with "
          "<font face='Courier'>default_rng(576)</font> so the transcript reproduces exactly. "
          "Values in [0,1) keep the threshold rows (<font face='Courier'>a &gt; 0.5</font>) "
          "non-trivial.", BODY),
      Paragraph("Rows that cannot be run exactly as printed", H2)]

for label, why in [
    ("28, 30", "the table's command assigns in place; it is run on a copy so later rows still see "
     "the original <font face='Courier'>a</font>"),
    ("57, 58", "the bare <font face='Courier'>logical_and</font> is called as "
     "<font face='Courier'>np.logical_and</font>"),
    ("59, 60", "bitwise operators need integers, so an integer pair "
     "<font face='Courier'>ai, bi</font> is used"),
    ("65", "the table gives prose, not a command; written out as "
     "<font face='Courier'>linalg.solve(a.T, b.T).T</font>"),
    ("67, 73", "Cholesky and conjugate gradients need positive definiteness, so "
     "<font face='Courier'>spd = a @ a.T + 5*np.eye(5)</font>"),
    ("80", "<font face='Courier'>signal.resample</font> needs an integer sample count, so "
     "<font face='Courier'>int(np.ceil(...))</font>"),
    ("82", "<font face='Courier'>a</font> has no singleton dimensions, so a 1&times;3&times;1&times;4 "
     "array is used instead"),
]:
    S.append(Paragraph(f"<b>Row {label}</b> &mdash; {why}.", NOTE))

S += [Spacer(1, 7),
      Paragraph("Task 2 &mdash; IPython transcript", H2)]

_s2 = shot("task2_ipython")
if _s2 is not None:
    S += [_s2, Spacer(1, 7)]

transcript = read("outputs/task2_transcript.txt")
chunks, buf = [], []
for line in transcript.split("\n"):
    if line.startswith("--- Row") and buf:
        chunks.append("\n".join(buf)); buf = []
    buf.append(line)
chunks.append("\n".join(buf))
for ch in chunks:
    if ch.strip():
        S.append(code(ch))

# ------------------------------------------------------------ tasks 3, 4 ---
S += [PageBreak(),
      Paragraph("3&nbsp;&nbsp;Plotting &mdash; the given script", H1),
      code("import matplotlib.pyplot as plt\n"
           "plt.plot([1,2,3,4], [1,2,7,14])\n"
           "plt.axis([0, 6, 0, 20])\n"
           "plt.show()", CODE_BIG),
      Paragraph(
          "Run verbatim in IPython. To capture it, the script was run under the "
          "<font face='Courier'>Agg</font> backend with a trailing "
          "<font face='Courier'>plt.savefig</font>; under <font face='Courier'>Agg</font>, "
          "<font face='Courier'>plt.show()</font> warns that it cannot display interactively, "
          "which does not affect the figure.", BODY),
      shot("task3_figure", 4.3) or Image("figures/task3.png", width=4.1 * inch, height=3.01 * inch),

      Paragraph("4&nbsp;&nbsp;Plotting &mdash; a figure of my own", H1),
      Paragraph(
          "The three classical activation functions against their derivatives. The right panel is the "
          "point: sigmoid's derivative peaks at 0.25 and decays to zero in both tails, so a product of "
          "many such factors vanishes with depth, while ReLU's is exactly 1 on the positive half-line.", BODY),
      shot("task4_figure") or Image("figures/task4.png", width=6.6 * inch, height=2.77 * inch),
      Spacer(1, 6),
      code(read("task4_plot.py"), CODE)]

# ------------------------------------------------------------ tasks 5, 6 ---
S += [KeepTogether([
      Paragraph("5&nbsp;&nbsp;Version Control &mdash; GitHub account", H1),
      code(f"{GH_USER}   ->   https://github.com/{GH_USER}", CODE_BIG)]),
      KeepTogether([
      Paragraph("6&nbsp;&nbsp;IDE project, pushed to GitHub", H1),
      Paragraph(
          "Created as a Python project against the Anaconda interpreter from Task 1, committed with "
          "git and pushed as a public repository. This assignment is the "
          "<font face='Courier'>Assignment0/</font> folder.", BODY),
      code(REPO_URL, CODE_BIG)])]

doc = SimpleDocTemplate(OUT, pagesize=letter,
                        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                        topMargin=0.8 * inch, bottomMargin=0.7 * inch,
                        title=f"ELEC 576 Assignment 0 - {STUDENT}", author=STUDENT)
doc.build(S)
print(f"wrote {OUT}")
