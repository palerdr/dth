"""Drop the Handkerchief tablebase: one certified game value per state class, produced by a single
backward pass over the potential layers phi = 1200..0 (reference.md)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

# ---- rules and sizes ---------------------------------------------------------------------------
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
DEAD_PHI = 301                # potential offset (and stored TTD) of a dead profile
MAX_PHI = MAX_ST + DEAD_PHI   # profile potential 0..600
MAX_LAYER = 2 * MAX_PHI       # class potential 0..1200
MAX_SADDLE_GAP = 1e-6         # the certificate gate
PURE, SUPPORT, LP = 0, 1, 2   # solver rungs, stored in K
UNSOLVED = 255                # K sentinel; V uses NaN
CHUNK = 32768                 # classes per batched gather/solve
RECHECK_SAMPLES = 1200        # classes re-derived independently in finalize

ANCHORS = [  # ((s_c, t_c, s_d, t_d), certified value), reference.md section 7
    ((0, 0, 0, 0), 0.08985007280951046),
    ((240, 0, 240, 0), 0.3372132166291093),
    ((10, 60, 200, 0), -0.7944428916469297),
    ((150, 90, 30, 120), 0.7244093036356785),
    ((250, 300, 40, 0), 0.9981152817381969),
    ((100, 140, 100, 140), 0.1877386378276193),
]


@dataclass
class State:
    dropper_st: int
    dropper_ttd: int
    checker_st: int
    checker_ttd: int


@dataclass
class ProfileTable:
    alive_id: np.ndarray          # (MAX_ST + 1, MAX_TTD + 1) int32; -1 where not alive / off-domain
    st: np.ndarray                # (N,) int32
    ttd: np.ndarray               # (N,) int32; DEAD_PHI for dead profiles
    alive: np.ndarray             # (N,) bool
    phi: np.ndarray               # (N,) int32 = st + ttd (dead: st + DEAD_PHI)
    rev: np.ndarray               # (N,) float64 revival probability (0 for dead)
    fail_child: np.ndarray        # (N,) int32; failed-check child id, or WIN
    success_children: np.ndarray  # (N, LAGS) int32; [p, lag - 1] = child id, or WIN
    bucket: list[np.ndarray]      # MAX_PHI + 1 int32 arrays; bucket[v] = ascending ids with phi == v


@dataclass
class SolveResult:
    maximin: float      # lower bound on the value: the Checker's best reply to the Dropper's mix
    minimax: float      # upper bound on the value: the Dropper's best reply to the Checker's mix
    kind: int           # PURE / SUPPORT / LP: the rung that produced the bounds
    saddle_gap: float = field(init=False)
    certified: bool = field(init=False)
    value: float | None = field(init=False)   # the certificate midpoint; None when uncertified

    def __post_init__(self):
        self.maximin = float(self.maximin)
        self.minimax = float(self.minimax)
        self.saddle_gap = self.minimax - self.maximin
        self.certified = bool(np.isfinite(self.saddle_gap) and self.saddle_gap <= MAX_SADDLE_GAP)
        self.value = 0.5 * (self.maximin + self.minimax) if self.certified else None


# ---- step 1: the quotient and its rule tables ---------------------------------------------------
def survives_injection(s: int, t: int) -> bool:
    dose = s + DOSE
    return dose <= MAX_ST and dose + t <= MAX_TTD


def revival_probability(s: int, t: int) -> float:
    if not survives_injection(s, t):
        return 0.0
    return 0.95 * (1 - (s / 240.0)) * (0.75 ** (t / 60.0))


def build_table() -> ProfileTable:
    alive_id = np.full((MAX_ST + 1, MAX_TTD + 1), -1, dtype=np.int32)
    st = np.empty(N, dtype=np.int32)
    ttd = np.empty(N, dtype=np.int32)

    # alive profiles in the normative order: TTD ascending over {0} then 60..300, ST ascending inside
    next_id = 0
    for t in [0] + list(range(60, MAX_TTD + 1)):
        for s in range(MAX_ST + 1):
            if survives_injection(s, t):
                alive_id[s, t] = next_id
                st[next_id] = s
                ttd[next_id] = t
                next_id += 1
    assert next_id == N_ALIVE, "Wrong number of Alive Profiles indexed into profile table"

    # dead sentinels, by ST
    st[N_ALIVE:] = np.arange(N_DEAD, dtype=np.int32)
    ttd[N_ALIVE:] = DEAD_PHI

    alive = np.arange(N) < N_ALIVE
    phi = st + ttd

    succ = np.empty((N, LAGS), dtype=np.int32)
    fail = np.empty(N, dtype=np.int32)
    rev = np.zeros(N, dtype=np.float64)

    for p in range(N):
        s, t = int(st[p]), int(ttd[p])
        is_alive = p < N_ALIVE
        for lag in range(1, LAGS + 1):
            grown = s + lag
            if grown >= CAPACITY:                                # capacity reached: mover wins
                succ[p, lag - 1] = WIN
            elif is_alive and survives_injection(grown, t):
                succ[p, lag - 1] = alive_id[grown, t]
            else:
                succ[p, lag - 1] = N_ALIVE + grown               # dead sentinel at ST grown
        if is_alive:
            rev[p] = revival_probability(s, t)
            new_ttd = s + t + DOSE
            fail[p] = alive_id[0, new_ttd] if survives_injection(0, new_ttd) else N_ALIVE  # dead sentinel at ST 0
        else:
            fail[p] = WIN                                        # a dead Checker loses every failed check

    bucket = [np.flatnonzero(phi == v).astype(np.int32) for v in range(MAX_PHI + 1)]

    # the potential must strictly increase on every transition; this is what makes the layer sweep sound
    has_child = succ != WIN
    assert (phi[succ[has_child]] > np.broadcast_to(phi[:, None], succ.shape)[has_child]).all(), \
        "phi must strictly increase on successful checks"
    has_child = fail != WIN
    assert (phi[fail[has_child]] > phi[has_child]).all(), "phi must strictly increase on failed checks"

    return ProfileTable(
        alive_id=alive_id, st=st, ttd=ttd, alive=alive, phi=phi, rev=rev,
        fail_child=fail, success_children=succ, bucket=bucket,
    )


def profile(s: int, t: int, table: ProfileTable) -> int:
    if not survives_injection(s, t):
        return N_ALIVE + s                          # dead: TTD is discarded exactly
    pid = int(table.alive_id[s, t])
    if pid == -1:
        raise IndexError("alive TTD in 1..59 is off-domain")
    return pid


def encode_state(state: State, table: ProfileTable) -> int:
    return profile(state.checker_st, state.checker_ttd, table) * N + profile(state.dropper_st, state.dropper_ttd, table)


def decode_class(c: int) -> tuple[int, int]:
    return divmod(int(c), N)


# ---- step 2: value storage, layer enumeration, gather ------------------------------------------
def values_array() -> np.ndarray:
    V = np.full((N, N + 1), np.nan, dtype=np.float64)
    V[:, WIN] = -1.0
    return V


def kinds_array() -> np.ndarray:
    return np.full((N, N), UNSOLVED, dtype=np.uint8)


def class_values(pc: int, pd: int, V: np.ndarray, table: ProfileTable) -> tuple[np.ndarray, float]:
    """Scalar oracle, written as reference.md section 5.3: the 60 successful-check values and the failed-check value."""
    s = np.empty(LAGS, dtype=np.float64)
    for lag in range(1, LAGS + 1):
        child = int(table.success_children[pc, lag - 1])
        s[lag - 1] = 1.0 if child == WIN else -V[pd, child]
    child = int(table.fail_child[pc])
    if child == WIN:
        f = 1.0
    else:
        p = float(table.rev[pc])
        f = p * (-V[pd, child]) + (1.0 - p)
    if not (np.isfinite(s).all() and np.isfinite(f)):
        raise RuntimeError(f"unsolved child read for class ({pc}, {pd}): schedule bug")
    return s, float(f)


def layer_pairs(P: int, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:
    """All classes (pc, pd) with phi[pc] + phi[pd] == P, pd-major inside each bucket rectangle."""
    pcs, pds = [], []
    for a in range(max(0, P - MAX_PHI), min(MAX_PHI, P) + 1):
        bc, bd = table.bucket[a], table.bucket[P - a]
        if bc.size and bd.size:
            pds.append(np.repeat(bd, bc.size))
            pcs.append(np.tile(bc, bd.size))
    if not pcs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(pcs), np.concatenate(pds)


def gather(pcs: np.ndarray, pds: np.ndarray, V: np.ndarray, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:
    """Batched continuation values: S (B, 60) and F (B,). Branch-free thanks to the WIN column and rev == 0 for dead."""
    flat = V.reshape(-1)                      # V is C-contiguous; row pd starts at pd * (N + 1)
    rows = pds.astype(np.int64) * V.shape[1]
    S = -np.take(flat, rows[:, None] + table.success_children[pcs])
    rev = table.rev[pcs]
    F = rev * (-np.take(flat, rows + table.fail_child[pcs])) + (1.0 - rev)
    if np.isnan(S).any() or np.isnan(F).any():
        raise RuntimeError("unsolved child read: schedule bug")
    return S, F


def iter_chunks(P: int, table: ProfileTable, chunk: int = CHUNK):
    pcs, pds = layer_pairs(P, table)
    for lo in range(0, pcs.size, chunk):
        hi = min(lo + chunk, pcs.size)
        yield lo, hi, pcs[lo:hi], pds[lo:hi]


# ---- step 3: the solver ladder, batched -------------------------------------------------------
def rung1_batch(S: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure saddle scan (reference 5.4) in closed form: (maximin, minimax) per row."""
    s0 = S[:, 0]
    maximin = np.maximum(S.min(axis=1), np.minimum(F, s0))
    minimax = np.minimum(S.max(axis=1), np.maximum(F, s0))
    return maximin, minimax


def toeplitz_matvec(S: np.ndarray, F: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(M q) per row for M[d, c] = S[c - d] (c >= d) else F:  Mq[i] = F * sum_{j<i} q[j] + sum_{k>=i} S[k-i] q[k]."""
    n = S.shape[1]
    L = 1 << (2 * n - 1).bit_length()          # zero-padded length so the circular correlation is exact
    corr = np.fft.irfft(np.fft.rfft(q, L, axis=1) * np.conj(np.fft.rfft(S, L, axis=1)), L, axis=1)[:, :n]
    cum = np.cumsum(q, axis=1)
    corr[:, 1:] += F[:, None] * cum[:, :-1]
    return corr


def rung2_batch(S: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-support equalizer (reference 5.5) per row: (maximin, minimax), NaN where unusable.

    With d0 = S[0] - F and dS = diff(S), the recurrence r[0] = 1, r[k] = -sum_{m<k} dS[m] r[k-1-m] / d0
    equalizes both sides at once: the Dropper's mix is r / sum(r), the Checker's is reverse(r) / sum(r),
    and (M q)[i] == (p M)[n-1-i], so max/min of M q are the two-sided certificate bounds.
    """
    B, n = S.shape
    d0 = S[:, 0] - F
    usable = np.abs(d0) >= 1e-12
    inv = np.zeros(B)
    np.divide(1.0, d0, out=inv, where=usable)
    dS_T = np.ascontiguousarray(np.diff(S, axis=1).T)   # (n - 1, B): contiguous along the batch
    r_T = np.empty((n, B))
    r_T[0] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n):
            r_T[k] = -np.einsum("mb,mb->b", dS_T[:k], r_T[k - 1::-1]) * inv
        r = r_T.T
        usable &= np.isfinite(r).all(axis=1)
        q = np.maximum(r[:, ::-1], 0.0)
        qs = q.sum(axis=1)
        usable &= qs > 0
        q = q / np.where(usable, qs, 1.0)[:, None]
        q[~usable] = 0.0
        Mq = toeplitz_matvec(S, F, q)
        maximin = np.where(usable, Mq.min(axis=1), np.nan)
        minimax = np.where(usable, Mq.max(axis=1), np.nan)
    return maximin, minimax


# ---- step 3: the solver ladder, scalar (oracle for tests and the finalize recheck) -------------
def _M_dot(s: np.ndarray, f: float, q: np.ndarray) -> np.ndarray:
    n = len(q)
    Mq = np.zeros(n)
    Mq[1:] = f * np.cumsum(q)[:-1]
    for j in range(n):
        Mq[:n - j] += s[j] * q[j:]
    return Mq


def full_matrix(s: np.ndarray, f: float) -> np.ndarray:
    n = len(s)
    g = np.arange(n)[None, :] - np.arange(n)[:, None]
    return np.where(g >= 0, s[g.clip(min=0)], f)


def try_rung1(s: np.ndarray, f: float) -> SolveResult:
    """O(60) pure saddle."""
    s0, lo, hi = s[0], s.min(), s.max()
    return SolveResult(maximin=max(lo, min(f, s0)), minimax=min(hi, max(f, s0)), kind=PURE)


def try_rung2(s: np.ndarray, f: float) -> SolveResult | None:
    """Full-support equalizer, O(n^2)."""
    n = len(s)
    d0 = s[0] - f
    dS = np.diff(s)
    if abs(d0) < 1e-12:
        return None
    r = np.zeros(n)
    r[0] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n):
            r[k] = -np.dot(dS[:k], r[k - 1::-1]) / d0
    if not np.all(np.isfinite(r)):
        return None
    q = np.clip(r[::-1], 0.0, None)
    if q.sum() <= 0:
        return None
    q /= q.sum()
    Mq = _M_dot(s, f, q)
    return SolveResult(maximin=Mq.min(), minimax=Mq.max(), kind=SUPPORT)


def try_rung3(s: np.ndarray, f: float) -> SolveResult:
    """LP residue: one LP for the Dropper's mix p; the Checker's mix q comes from the duals."""
    s, f = np.asarray(s, dtype=float), float(f)
    n = len(s)
    M = full_matrix(s, f)
    c = np.zeros(n + 1)
    c[n] = -1.0                                      # maximize v
    A_ub = np.hstack([-M.T, np.ones((n, 1))])        # v <= (p^T M)_c  for all c
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    bnds = [(0, None)] * n + [(None, None)]          # v free (the v = 0 knife-edge)

    best = None
    for method in ("highs-ds", "highs-ipm"):         # retries change the solver, never the gate
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0], bounds=bnds, method=method)
        if not res.success:
            continue
        p = np.clip(res.x[:n], 0, None)
        p /= p.sum()
        q = np.clip(-res.ineqlin.marginals, 0, None)
        if q.sum() <= 0:
            continue
        q /= q.sum()
        out = SolveResult(maximin=(p @ M).min(), minimax=(M @ q).max(), kind=LP)
        if out.certified:
            return out
        if best is None or out.saddle_gap < best.saddle_gap:
            best = out
    return best if best is not None else SolveResult(maximin=np.nan, minimax=np.nan, kind=LP)


def solve_class(s: np.ndarray, f: float) -> SolveResult:
    """The ladder for one class (reference 5.7)."""
    res = try_rung1(s, f)
    if res.certified:
        return res
    res = try_rung2(s, f)
    if res is not None and res.certified:
        return res
    res = try_rung3(s, f)
    if not res.certified:
        raise RuntimeError(f"uncertified class (gap {res.saddle_gap}) - aborting build")
    return res


# ---- step 4: one chunk through the ladder ----------------------------------------------------
def solve_chunk(S: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values and kinds for a chunk: batched rungs 1-2, then the LP worklist for whatever is left."""
    B = S.shape[0]
    values = np.empty(B, dtype=np.float64)
    kinds = np.empty(B, dtype=np.uint8)

    lo1, hi1 = rung1_batch(S, F)
    pure = (hi1 - lo1) <= MAX_SADDLE_GAP
    values[pure] = 0.5 * (lo1[pure] + hi1[pure])
    kinds[pure] = PURE

    if not pure.all():
        lo2, hi2 = rung2_batch(S, F)
        with np.errstate(invalid="ignore"):
            support = ~pure & ((hi2 - lo2) <= MAX_SADDLE_GAP)   # NaN gaps compare False
        values[support] = 0.5 * (lo2[support] + hi2[support])
        kinds[support] = SUPPORT
        for i in np.flatnonzero(~(pure | support)):
            res = try_rung3(S[i], F[i])
            if not res.certified:
                raise RuntimeError(f"uncertified class in chunk row {i} (gap {res.saddle_gap}) - aborting build")
            values[i] = res.value
            kinds[i] = LP
    return values, kinds


# ---- step 5: layers, sweep, checkpoint, finalize -------------------------------------------------
def solve_layer(P: int, V: np.ndarray, K: np.ndarray, table: ProfileTable, chunk: int = CHUNK) -> tuple[int, np.ndarray]:
    counts = np.zeros(3, dtype=np.int64)
    n = 0
    for _lo, _hi, pc, pd in iter_chunks(P, table, chunk):
        S, F = gather(pc, pd, V, table)
        vals, kinds = solve_chunk(S, F)
        V[pc, pd] = vals
        K[pc, pd] = kinds
        counts += np.bincount(kinds, minlength=3)
        n += pc.size
    return n, counts


@dataclass
class Checkpoint:
    next: int = MAX_LAYER                 # the next layer to solve; -1 when the sweep is done
    counts: list[int] = field(default_factory=lambda: [0, 0, 0])   # pure, support, lp so far
    complete: bool = False                # finalize + anchors passed
    elapsed: float = 0.0                  # build seconds so far


def _paths(build_dir: str) -> dict[str, str]:
    return {k: os.path.join(build_dir, f) for k, f in
            (("V", "V.npy"), ("K", "K.npy"), ("ckpt", "checkpoint.json"))}


def load_build(build_dir: str) -> tuple[np.ndarray, np.ndarray, Checkpoint]:
    p = _paths(build_dir)
    if not os.path.exists(p["ckpt"]):
        return values_array(), kinds_array(), Checkpoint()
    with open(p["ckpt"]) as fh:
        ck = Checkpoint(**json.load(fh))
    V = np.load(p["V"])
    K = np.load(p["K"])
    assert V.shape == (N, N + 1) and K.shape == (N, N), "checkpoint arrays have the wrong shape"
    return np.ascontiguousarray(V), np.ascontiguousarray(K), ck


def save_build(build_dir: str, V: np.ndarray, K: np.ndarray, ck: Checkpoint) -> None:
    """Arrays first, checkpoint last, each via atomic replace: a crash mid-save leaves an older, consistent state."""
    os.makedirs(build_dir, exist_ok=True)
    p = _paths(build_dir)
    for key, arr in (("V", V), ("K", K)):
        tmp = p[key] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, p[key])
    tmp = p["ckpt"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ck.__dict__, fh)
    os.replace(tmp, p["ckpt"])


def recheck(c: int, V: np.ndarray, table: ProfileTable) -> SolveResult:
    """Independent re-derivation of one class from its stored children (reference 5.10)."""
    pc, pd = decode_class(c)
    s, f = class_values(pc, pd, V, table)
    return solve_class(s, f)


def finalize(V: np.ndarray, K: np.ndarray, table: ProfileTable, log=print) -> None:
    core = V[:, :N]
    assert not np.isnan(core).any(), "unsolved classes remain"
    assert (np.abs(core) <= 1 + 1e-9).all(), "a value left [-1, 1]"
    assert (K <= LP).all(), "a class has no rung recorded"
    worst = 0.0
    for c in np.linspace(0, CLASSES - 1, RECHECK_SAMPLES).astype(np.int64):
        pc, pd = decode_class(c)
        err = abs(recheck(c, V, table).value - V[pc, pd])
        worst = max(worst, err)
        assert err <= MAX_SADDLE_GAP, f"recheck mismatch at class {c}: {err}"
    log(f"finalize: {RECHECK_SAMPLES} rechecks ok, worst |diff| = {worst:.3e}")


def verify_anchors(V: np.ndarray, table: ProfileTable) -> list[tuple[tuple, float, float, bool]]:
    out = []
    for (sc, tc, sd, td), expected in ANCHORS:
        c = encode_state(State(dropper_st=sd, dropper_ttd=td, checker_st=sc, checker_ttd=tc), table)
        pc, pd = decode_class(c)
        got = float(V[pc, pd])
        out.append(((sc, tc, sd, td), expected, got, abs(got - expected) <= MAX_SADDLE_GAP))
    return out


def sweep(build_dir: str = "build", max_layers: int | None = None, stop_at: int = 0,
          checkpoint_minutes: float = 10.0, chunk: int = CHUNK, table: ProfileTable | None = None,
          save: bool = True, log=print) -> tuple[np.ndarray, np.ndarray, Checkpoint]:
    """Solve layers from the checkpoint's `next` down to `stop_at`; finalize when the sweep reaches -1.
    `save=False` keeps everything in memory (tests)."""
    table = table or build_table()
    V, K, ck = load_build(build_dir)
    if ck.complete:
        log("build already complete")
        return V, K, ck
    t_start = time.perf_counter() - ck.elapsed
    t_saved = time.perf_counter()
    done_this_session = 0
    layer_total = 0
    log(f"sweep: resuming at layer {ck.next}" if ck.next < MAX_LAYER else "sweep: fresh build")
    while ck.next >= stop_at and (max_layers is None or done_this_session < max_layers):
        P = ck.next
        t0 = time.perf_counter()
        n, counts = solve_layer(P, V, K, table, chunk)
        dt = time.perf_counter() - t0
        ck.counts = [int(a + b) for a, b in zip(ck.counts, counts)]
        ck.next = P - 1
        ck.elapsed = time.perf_counter() - t_start
        done_this_session += 1
        layer_total += n
        if n:
            log(f"P={P:4d} n={n:8d} pure={counts[0]:7d} sup={counts[1]:8d} lp={counts[2]:6d} "
                f"{dt:7.2f}s {n / dt:9.0f}/s | elapsed {ck.elapsed:8.1f}s")
        if save and (time.perf_counter() - t_saved) / 60.0 >= checkpoint_minutes:
            save_build(build_dir, V, K, ck)
            t_saved = time.perf_counter()
            log(f"checkpoint saved at next={ck.next}")
    if ck.next < 0 and not ck.complete:
        finalize(V, K, table, log)
        results = verify_anchors(V, table)
        for state, expected, got, ok in results:
            log(f"anchor {state}: expected {expected:.16f} got {got:.16f} {'ok' if ok else 'MISMATCH'}")
        assert all(r[3] for r in results), "anchor mismatch"
        ck.complete = True
        log(f"build complete: pure={ck.counts[0]} support={ck.counts[1]} lp={ck.counts[2]} "
            f"total={sum(ck.counts)} in {ck.elapsed:.0f}s")
    if save:
        save_build(build_dir, V, K, ck)
    return V, K, ck


# ---- CLI ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DTH tablebase builder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build or resume the tablebase")
    b.add_argument("layers", nargs="?", type=int, default=None, help="bound this session to that many layers")
    b.add_argument("--build-dir", default="build")
    b.add_argument("--checkpoint-minutes", type=float, default=10.0)
    b.add_argument("--chunk", type=int, default=CHUNK)
    v = sub.add_parser("verify", help="check the anchors of a finished build")
    v.add_argument("--build-dir", default="build")
    s = sub.add_parser("status", help="show the checkpoint")
    s.add_argument("--build-dir", default="build")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.cmd == "build":
        sweep(args.build_dir, max_layers=args.layers, checkpoint_minutes=args.checkpoint_minutes,
              chunk=args.chunk, log=log)
        return 0
    if args.cmd == "verify":
        V, _K, ck = load_build(args.build_dir)
        if not ck.complete:
            log(f"build not complete (next layer {ck.next})")
        ok = True
        for state, expected, got, good in verify_anchors(V, build_table()):
            log(f"anchor {state}: expected {expected:.16f} got {got:.16f} {'ok' if good else 'MISMATCH'}")
            ok &= good
        return 0 if ok else 1
    if args.cmd == "status":
        p = _paths(args.build_dir)
        if not os.path.exists(p["ckpt"]):
            log("no checkpoint")
            return 0
        with open(p["ckpt"]) as fh:
            log(json.dumps(json.load(fh), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
