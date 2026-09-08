"""Write the two data files the paper's pgfplots figures read: the layer widths and the root strategies.
Needs a finished build in build/V.npy."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dth  # noqa: E402

OUT = ROOT / "paper" / "build" / "figures"


def equalizer(s, f):
    """The clipped full-support equilibrium of the stage matrix (paper, Proposition 6.3): the Dropper's
    mix, the Checker's mix, and the certificate bounds (lower, upper)."""
    d0, dS = s[0] - f, np.diff(s)
    r = np.zeros(60)
    r[0] = 1.0
    for k in range(1, 60):
        r[k] = -np.dot(dS[:k], r[k - 1::-1]) / d0
    r = np.clip(r, 0.0, None)
    r /= r.sum()
    g = np.arange(60)[None, :] - np.arange(60)[:, None]
    M = np.where(g >= 0, s[g.clip(min=0)], f)
    return r, r[::-1], (r @ M).min(), (M @ r[::-1]).max()


def main() -> None:
    T = dth.build_table()
    V = np.load(ROOT / "build" / "V.npy", mmap_mode="r")
    OUT.mkdir(parents=True, exist_ok=True)

    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    widths = np.convolve(sizes, sizes)
    assert widths.size == 1201 and widths.sum() == dth.CLASSES
    with (OUT / "layer_widths.dat").open("w") as fh:
        fh.write("potential classes\n")
        fh.writelines(f"{P} {int(w)}\n" for P, w in enumerate(widths))

    root = dth.profile(0, 0, T)
    s, p = -V[root, T.succ[root]], float(T.rev[root])
    f = p * (-float(V[root, T.fail[root]])) + (1.0 - p)
    drop, check, lo, hi = equalizer(s, f)
    assert hi - lo <= dth.MAX_SADDLE_GAP and abs(0.5 * (lo + hi) - V[root, root]) <= dth.MAX_SADDLE_GAP
    with (OUT / "root_strategies.dat").open("w") as fh:
        fh.write("action drop check\n")
        fh.writelines(f"{a + 1} {drop[a]:.10f} {check[a]:.10f}\n" for a in range(60))


if __name__ == "__main__":
    main()
