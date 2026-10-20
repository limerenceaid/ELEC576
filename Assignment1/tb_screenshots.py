import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

PORT = 6131
SCALE = 2
WIDTH = 1700

groups = {
    'baseline': ['baseline'],
    'act': ['baseline', 'act_tanh', 'act_sigmoid', 'act_leaky_relu'],
    'init': ['sgd', 'sgd_xavier', 'sgd_kaiming'],
    'opt': ['baseline', 'sgd', 'momentum', 'adagrad'],
}

shots = [
    ('a_train_loss', 'baseline', 'scalars', r'^train/loss$'),
    ('b_test', 'baseline', 'scalars', r'^test/(loss|error)$'),
    ('b_hist_params', 'baseline', 'histograms', r'^(weights|biases)/'),
    ('b_hist_netinput', 'baseline', 'histograms', r'^net_input/'),
    ('b_hist_act', 'baseline', 'histograms', r'^(activation|maxpool)/'),
    ('b_stats_params', 'baseline', 'scalars', r'^(weights|biases)/fc1/'),
    ('b_stats_conv2', 'baseline', 'scalars', r'^(net_input|activation|maxpool)/conv2/'),
    ('b_std_params', 'baseline', 'scalars', r'^(weights|biases)/.*/std$'),
    ('b_std_tensors', 'baseline', 'scalars', r'^(net_input|activation|maxpool)/.*/std$'),
    ('c_act', 'act', 'scalars', r'^(train/loss|test/error)$'),
    ('c_act_hist', 'act', 'histograms', r'^activation/fc1$'),
    ('c_init', 'init', 'scalars', r'^(train/loss|test/error)$'),
    ('c_init_std', 'init', 'scalars', r'^activation/(conv1|conv2|fc1)/std$'),
    ('c_opt', 'opt', 'scalars', r'^(train/loss|test/error)$'),
]


def crop(path, n_runs):
    img = Image.open(path).convert('RGB')
    a = np.asarray(img).astype(int)
    side = 345 * SCALE
    nonwhite = (a < 250).any(axis=2)
    y = 190 * SCALE
    while y < a.shape[0] and nonwhite[y, side + 30 * SCALE:a.shape[1] - 30 * SCALE].mean() < 0.9:
        y += 1
    box = nonwhite[175 * SCALE:y - 2 * SCALE, side + 20 * SCALE:a.shape[1] - 20 * SCALE]
    ys = np.where(box.any(axis=1))[0]
    xs = np.where(box.any(axis=0))[0]
    card = img.crop((side + 20 * SCALE + xs[0] - 8 * SCALE, 175 * SCALE + ys[0] - 4 * SCALE,
                     side + 20 * SCALE + xs[-1] + 8 * SCALE, 175 * SCALE + ys[-1] + 4 * SCALE))
    if n_runs <= 1:
        return card
    runs = img.crop((28 * SCALE, 474 * SCALE, 250 * SCALE, (474 + 28 * n_runs) * SCALE))
    out = Image.new('RGB', (runs.width + card.width, max(runs.height, card.height)), 'white')
    out.paste(card, (0, 0))
    out.paste(runs, (card.width, 8 * SCALE))
    return out


def main():
    logdir, outdir = os.path.abspath(sys.argv[1]), sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='chrome')
        for port, (group, runs) in enumerate(groups.items(), PORT):
            tmp = tempfile.mkdtemp()
            for r in runs:
                os.symlink(os.path.join(logdir, r), os.path.join(tmp, r))
            tb = subprocess.Popen(['tensorboard', '--logdir', tmp, '--port', str(port), '--samples_per_plugin', 'histograms=100'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(12)
            try:
                for name, g, dash, tags in shots:
                    if g != group:
                        continue
                    smooth = '0.6' if 'train/loss' in tags else '0'
                    frag = '%s&tagFilter=%s&_smoothingWeight=%s' % (dash, urllib.parse.quote(tags, safe=''), smooth)
                    page = browser.new_page(viewport={'width': WIDTH, 'height': 1800}, device_scale_factor=SCALE)
                    page.goto('http://localhost:%d/?darkMode=false#%s' % (port, frag))
                    page.wait_for_timeout(8000)
                    if dash == 'scalars':
                        box = page.get_by_role('checkbox', name='Ignore outliers in chart scaling')
                        if box.is_checked():
                            box.click()
                        page.wait_for_timeout(3000)
                    raw = os.path.join(outdir, name + '_raw.png')
                    page.screenshot(path=raw)
                    crop(raw, len(runs) if dash == 'scalars' else 1).save(os.path.join(outdir, name + '.png'))
                    os.remove(raw)
                    page.close()
                    print(name, flush=True)
            finally:
                tb.terminate()
                tb.wait()
                shutil.rmtree(tmp)
        browser.close()


if __name__ == '__main__':
    main()
