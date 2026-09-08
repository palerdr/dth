"""We solve Drop the Handkerchief in one pass over potential layers phi = 1200..0.
We use 16-class C batches and check stage-game certificates at a gap of 1e-6.
You can find the game rules and error bounds in paper/dth.tex."""

from __future__ import annotations
import ctypes
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

# GAME CONSTANTS AND TABLEBASE INDEXING
MAX_ST = 299                  # squandered time 0..299
MAX_TTD = 300                 # accrued toxin time 0..300
CAPACITY = 300                # a successful check that grows ST to >= CAPACITY wins for the mover
DOSE = 60                     # toxin time taken on a failed check
LAGS = 60                     # actions 1..60; a successful check elapses lag = c - d + 1 seconds
N_ALIVE = 16711               # alive profiles: ids 0..16710 in the normative order
N_DEAD = 300                  # dead sentinels: id N_ALIVE + st
N = N_ALIVE + N_DEAD          # 17011 profiles
CLASSES = N * N               # 289,374,121 state classes; class (pc, pd) is V[pc, pd]
WIN = N                       # child column meaning "the mover wins": V[:, WIN] == -1.0, so -V reads +1
DEAD_PHI = 301                # potential offset of a dead profile
MAX_PHI = MAX_ST + DEAD_PHI   # profile potential 0..600
MAX_LAYER = 2 * MAX_PHI       # class potential 0..1200
MAX_SADDLE_GAP = 1e-6         # the certificate gate (GAP in kernel.c)

ANCHORS = [  # ((s_c, t_c, s_d, t_d), certified value)
    ((0, 0, 0, 0), 0.08985007280951046),
    ((240, 0, 240, 0), 0.3372132166291093),
    ((10, 60, 200, 0), -0.7944428916469297),
    ((150, 90, 30, 120), 0.7244093036356785),
    ((250, 300, 40, 0), 0.9981152817381969),
    ((100, 140, 100, 140), 0.1877386378276193),
]


@dataclass
class ProfileTable:
    alive_id: np.ndarray    # (MAX_ST + 1, MAX_TTD + 1) int32; -1 where not alive / off-domain
    st: np.ndarray          # (N,) int32
    ttd: np.ndarray         # (N,) int32; DEAD_PHI for dead profiles
    phi: np.ndarray         # (N,) int32 = st + ttd
    rev: np.ndarray         # (N,) float64 revival probability (0 for dead)
    fail: np.ndarray        # (N,) int32; failed-check child id, or WIN
    succ: np.ndarray        # (N, LAGS) int32; [p, lag - 1] = child id, or WIN
    bucket: list[np.ndarray]  # MAX_PHI + 1 int32 arrays; bucket[v] = ascending ids with phi == v


# ---- profile tables ----------------------------------------------------------------------------
def survives_injection(s: int, t: int) -> bool:
    dose = s + DOSE
    return dose <= MAX_ST and dose + t <= MAX_TTD


def revival_probability(s: int, t: int) -> float:
    if not survives_injection(s, t):
        return 0.0
    return 0.95 * (1 - (s / 240.0)) * (0.75 ** (t / 60.0))


def build_table() -> ProfileTable:
    """Profiles in the normative order (TTD ascending over {0} then 60..300, ST ascending inside each TTD),
    every transition table, and the phi buckets."""

    alive_id = np.full((MAX_ST + 1, MAX_TTD + 1), -1, dtype=np.int32)
    st = np.empty(N, dtype=np.int32)
    ttd = np.empty(N, dtype=np.int32)
    next_id = 0
    for t in [0] + list(range(DOSE, MAX_TTD + 1)):
        for s in range(MAX_ST + 1):
            if survives_injection(s, t):
                alive_id[s, t] = next_id
                st[next_id], ttd[next_id] = s, t
                next_id += 1

    assert next_id == N_ALIVE, "wrong number of alive profiles"

    st[N_ALIVE:] = np.arange(N_DEAD, dtype=np.int32)
    ttd[N_ALIVE:] = DEAD_PHI
    phi = st + ttd

    succ = np.empty((N, LAGS), dtype=np.int32)
    fail = np.empty(N, dtype=np.int32)
    rev = np.zeros(N, dtype=np.float64)
    for p in range(N):
        s, t, alive = int(st[p]), int(ttd[p]), p < N_ALIVE
        for lag in range(1, LAGS + 1):
            grown = s + lag
            if grown >= CAPACITY:
                succ[p, lag - 1] = WIN
            elif alive and survives_injection(grown, t):
                succ[p, lag - 1] = alive_id[grown, t]
            else:
                succ[p, lag - 1] = N_ALIVE + grown
        if alive:
            rev[p] = revival_probability(s, t)
            new_ttd = s + t + DOSE
            fail[p] = alive_id[0, new_ttd] if survives_injection(0, new_ttd) else N_ALIVE
        else:
            fail[p] = WIN
    bucket = [np.flatnonzero(phi == v).astype(np.int32) for v in range(MAX_PHI + 1)]

    has_child = succ != WIN
    assert (phi[succ[has_child]] > np.broadcast_to(phi[:, None], succ.shape)[has_child]).all()
    has_child = fail != WIN
    assert (phi[fail[has_child]] > phi[has_child]).all()
    return ProfileTable(alive_id, st, ttd, phi, rev, fail, succ, bucket)


def profile(s: int, t: int, table: ProfileTable) -> int:
    """Profile id of (ST, TTD); dead profiles collapse to the sentinel for their ST."""
    if not (0 <= s <= MAX_ST and 0 <= t <= MAX_TTD):
        raise IndexError("profile outside ST 0..299 or TTD 0..300")
    if not survives_injection(s, t):
        return N_ALIVE + s
    pid = int(table.alive_id[s, t])
    if pid == -1:
        raise IndexError("alive TTD in 1..59 is off-domain")
    return pid


def layer_pairs(P: int, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:
    """All classes (pc, pd) with phi[pc] + phi[pd] == P, pd-major so neighbours read the same row of V."""
    pcs, pds = [], []
    for a in range(max(0, P - MAX_PHI), min(MAX_PHI, P) + 1):
        bc, bd = table.bucket[a], table.bucket[P - a]
        if bc.size and bd.size:
            pds.append(np.repeat(bd, bc.size))
            pcs.append(np.tile(bc, bd.size))
    if not pcs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(pcs), np.concatenate(pds)


# backwards sweep
def load_kernel() -> ctypes.CDLL:
    """Compile kernel.c next to this file when its .so is missing or stale, then load it."""
    here = os.path.dirname(os.path.abspath(__file__))
    src, lib = os.path.join(here, "kernel.c"), os.path.join(here, "_kernel.so")
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        with tempfile.TemporaryDirectory(dir=here) as tmp:
            compiled = os.path.join(tmp, "kernel.so")
            subprocess.run(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", compiled, src, "-lm"], check=True)
            os.replace(compiled, lib)
    k = ctypes.CDLL(lib)
    k.solve_chunk.restype = ctypes.c_long
    k.solve_chunk.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return k


def solve_layer(P: int, V: np.ndarray, table: ProfileTable, kernel, pool: ThreadPoolExecutor,
                threads: int) -> np.ndarray:
    """Solve layer P with the kernel on chunks in parallel (ctypes releases the GIL). Returns the counts
    by rung (pure, equalizer); raises when a class fails the certificate gate."""
    pcs, pds = layer_pairs(P, table)
    if pcs.size == 0:
        return np.zeros(2, dtype=np.int64)
    chunk = pcs.size if threads == 1 else 2048 * max(1, -(-pcs.size // (2048 * 4 * threads)))
    ptr = lambda a: a.ctypes.data  # noqa: E731

    def run(lo: int) -> np.ndarray:
        hi = min(lo + chunk, pcs.size)
        part = np.zeros(2, dtype=np.int64)
        failed = kernel.solve_chunk(hi - lo, ptr(pcs[lo:]), ptr(pds[lo:]), ptr(V), V.shape[1],
                                    ptr(table.succ), ptr(table.fail), ptr(table.rev), ptr(part))
        if failed == -3:
            raise RuntimeError(f"layer {P}: require round-to-nearest with gradual underflow")
        if failed == -2:
            raise RuntimeError(f"layer {P}: invalid child value or revival probability")
        if failed < 0:
            raise RuntimeError(f"layer {P}: a certified value left [-1, 1]")
        if failed:
            raise RuntimeError(f"layer {P}: {failed} classes failed the certificate gate")
        return part

    if pcs.size <= chunk:
        return run(0)
    return np.sum(list(pool.map(run, range(0, pcs.size, chunk))), axis=0)


def build(table: ProfileTable, stop_at: int = 0, threads: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The sweep: layers MAX_LAYER down to stop_at. Returns V (NaN where unsolved) and the counts by rung."""
    threads = (os.cpu_count() or 1) if threads is None else threads
    if not 0 <= stop_at <= MAX_LAYER or threads < 1:
        raise ValueError("require stop_at in 0..1200 and threads >= 1")
    V = np.full((N, N + 1), np.nan, dtype=np.float64)
    V[:, WIN] = -1.0
    kernel = load_kernel()
    counts = np.zeros(2, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for P in range(MAX_LAYER, stop_at - 1, -1):
            counts += solve_layer(P, V, table, kernel, pool, threads)
    return V, counts


def verify(V: np.ndarray, table: ProfileTable) -> None:
    """We check the anchors and re-solve 1,200 classes with a NumPy certificate."""
    for (sc, tc, sd, td), expected in ANCHORS:
        got = V[profile(sc, tc, table), profile(sd, td, table)]
        if not abs(got - expected) <= MAX_SADDLE_GAP:
            raise RuntimeError(f"anchor {(sc, tc, sd, td)}: expected {expected}, got {got}")
    g = np.arange(LAGS)[None, :] - np.arange(LAGS)[:, None]
    for c in np.linspace(0, CLASSES - 1, 1200, dtype=np.int64):
        pc, pd = divmod(int(c), N)
        s = -V[pd, table.succ[pc]]
        p = table.rev[pc]
        f = p * (-V[pd, table.fail[pc]]) + (1.0 - p)
        M = np.where(g >= 0, s[g.clip(min=0)], f)
        lo, hi = M.min(axis=1).max(), M.max(axis=0).min()
        if hi - lo > MAX_SADDLE_GAP:
            delta = np.diff(s)
            r = np.zeros(LAGS)
            r[0] = 1.0
            with np.errstate(all="ignore"):
                for k in range(1, LAGS):
                    r[k] = -np.dot(delta[:k], r[k - 1::-1]) / (s[0] - f)
                if not np.isfinite(r).all():
                    raise RuntimeError(f"invalid equalizer in recheck for class {(pc, pd)}")
                r = np.maximum(r, 0.0)
                total = r.sum()
                if not (np.isfinite(total) and total > 0.0):
                    raise RuntimeError(f"invalid weight sum in recheck for class {(pc, pd)}")
                r /= total
            lo, hi = (r @ M).min(), (M @ r[::-1]).max()
        if not (np.isfinite([lo, hi]).all() and hi - lo <= MAX_SADDLE_GAP
                and abs(V[pc, pd] - (lo + hi) / 2) <= MAX_SADDLE_GAP):
            raise RuntimeError(f"certificate recheck failed for class {(pc, pd)}")


def main() -> None:
    table = build_table()
    t0 = time.perf_counter()
    V, counts = build(table)
    if counts.sum() != CLASSES:
        raise RuntimeError(f"{CLASSES - counts.sum()} classes unsolved")
    verify(V, table)
    print(f"{CLASSES} classes in {time.perf_counter() - t0:.1f}s: pure={counts[0]} equalizer={counts[1]}")
    for (sc, tc, sd, td), expected in ANCHORS:
        got = float(V[profile(sc, tc, table), profile(sd, td, table)])
        print(f"anchor {(sc, tc, sd, td)}: expected {expected:.16f} got {got:.16f}")
    os.makedirs("build", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="build") as tmp:
        path = os.path.join(tmp, "V.npy")
        np.save(path, V)
        os.replace(path, "build/V.npy")


if __name__ == "__main__":
    main()
