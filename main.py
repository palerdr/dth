CLASSES = 289374121
ACTIONS = 60
MAX_TTD = 300
MAX_ST = 299
FAILED_PENALTY = 60
ALIVE_PROFILES = 16710
DEAD_PROFILES = 300
TOTAL_PROFILES = 17010
BUCKETS = 600
MAX_SADDLE_GAP = 1e-6
DEAD_SENTINEL = 301
WIN = -1

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog

@dataclass
class State:
    dropper_st: int
    dropper_ttd: int
    checker_st: int
    checker_ttd: int

@dataclass
class ProfileTable:
    alive_id: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))
    dead_id: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))

    st: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))
    ttd: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))
    phi: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))
    rev: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))
    fail_child: np.ndarray = field(default_factory=lambda: np.zeros(TOTAL_PROFILES, dtype=np.int32))

    success_children: np.ndarray = field(default_factory=lambda: np.zeros((TOTAL_PROFILES, ACTIONS + 1), dtype=np.int32))

    bucket: np.ndarray = field(default_factory=lambda: np.zeros(BUCKETS, dtype=object))

@dataclass
class TransitionValues:
    success_values: np.ndarray = field(default_factory=lambda: np.zeros(CLASSES, dtype=np.float32))
    fail_value: np.float32 = field(default_factory=lambda: np.float32(0))
@dataclass
class SolveResult:
    minimax: np.float32
    maximin: np.float32
    saddle_gap: np.float32
    value: np.float32

def initial_table() -> ProfileTable:
    return ProfileTable()

def values_array() -> np.ndarray:
    return np.empty(CLASSES, dtype=np.float32)

def survives_injection(s:int , t:int) -> bool:
    dose = s + FAILED_PENALTY
    return dose <= MAX_ST and dose + t <= MAX_TTD

def revival_probability(s:int, t:int) -> float:
    if not survives_injection(s, t):
        return 0.0
    else:
        return 0.95 * (1 - ( s / 240.0)) * (0.75 ** ( t / 60.0))

def build_table() -> ProfileTable:
    table = initial_table()
    alive_id = np.full((MAX_ST, MAX_TTD), -1, dtype=np.int32)
    next_id = 0
    ttds = [0] + list(range(60,MAX_TTD + 1))
    sts = list(range(0, MAX_ST + 1))
    for t in ttds:
        for s in sts:
            if survives_injection(s,t):
                alive_id[s][t] = next_id
                table.st[next_id] = s
                table.ttd[next_id] = t
                next_id += 1

    assert (next_id == ALIVE_PROFILES), "Wrong number of Alive Profiles indexed into profile table"

    for s in sts:
        table.st[ALIVE_PROFILES + s] = s
        table.ttd[ALIVE_PROFILES + s] = DEAD_SENTINEL

    bucket_accumulator = [[] for _ in range(BUCKETS + 1)]

    for p in range(TOTAL_PROFILES):
        table.phi[p] = table.st[p] + table.ttd[p] if p <= ALIVE_PROFILES else table.st[p] + DEAD_SENTINEL

        for lag in range(1, ACTIONS):
            new_st = table.st[p] + lag
            if new_st >= MAX_TTD:
                table.success_children[p][lag] = WIN
            else:
                if p > ALIVE_PROFILES or not survives_injection(new_st, table.ttd[p]):
                    table.success_children[p][lag] = ALIVE_PROFILES + new_st
                else:
                    table.success_children[p][lag] = alive_id[new_st][table.ttd[p]]

        if p <= ALIVE_PROFILES:
            s, t = table.st[p], table.ttd[p]
            table.rev[p] = revival_probability(s, t)
            new_ttd = s + t + FAILED_PENALTY
            table.fail_child[p] = alive_id[0][new_ttd] if survives_injection(0, new_ttd) else ALIVE_PROFILES + new_ttd
        else:
            table.rev[p] = 0.0
            table.fail_child[p] = WIN

        v = table.phi[p]
        bucket_accumulator[v].append(p)

        for v in range(BUCKETS + 1):
            table.bucket[v] = np.array(bucket_accumulator[v], dtype=np.int32)

    return table

def profile(s:int, t:int, table: ProfileTable) -> int:
    if not survives_injection(s, t):
        return ALIVE_PROFILES + s
    else:
        if not table.alive_id[s][t]:
            raise IndexError("alive TTD in 1...59, is off-domain")
        else:
            return table.alive_id[s][t]

def encode_state(state: State, table: ProfileTable) -> int:
    return profile(state.checker_st, state.checker_ttd, table) * ALIVE_PROFILES + profile(state.dropper_st, state.dropper_ttd, table)

def class_values(pc:int , pd:int, V: np.ndarray , table: ProfileTable):
    t = TransitionValues()
    for lag in range(1, ACTIONS):
        if table.success_children[pc][lag] == WIN:
            t.success_values[lag] = 1
        else:
            t.success_values[lag] = -V[pd * TOTAL_PROFILES + table.success_children[pc][lag]]

    if table.fail_child[pc] == WIN:
        t.fail_value = np.float32(1)
    else:
        p = table.rev[pc]
        continuation_value = -V[pd * TOTAL_PROFILES + table.fail_child[pc]]
        t.fail_value = p * continuation_value + (1-p)
    
    if any([(not np.isfinite(x)) for x in t.success_values]) or not np.isfinite(t.fail_value):
        raise ValueError("Non finite game value computed")

    return t

def _M_dot(s, f, q):
    n= len(q)
    Mq = np.zeros(n)
    Mq[1:] = f * np.cumsum(q)[:-1]
    for j in range(n):
        Mq[:n-j] += s[j] * q[j:]
    return Mq

def try_rung1(t: TransitionValues) -> SolveResult | None:
    """O(60) pure saddle baby, 1 action prior"""
    s,f = t.success_values, t.fail_value
    s0, lo, hi = s[0], s.min(), s.max()

    maximin = np.float32(max(lo, min(f, s0)))
    minimax = np.float32(min(hi, max(f, s0)))
    saddle_gap = np.float32(minimax - maximin)

    if saddle_gap > MAX_SADDLE_GAP:
        return None

    return SolveResult(
        minimax=minimax,
        maximin=maximin,
        saddle_gap=saddle_gap,
        value=np.float32((minimax + maximin) * 0.5),
    )

def try_rung2(t: TransitionValues) -> SolveResult | None:
    """full support prior O(N^2) solve"""
    s, f = t.success_values, t.fail_value
    n = len(s)
    d0 = s[0] - f
    dS = np.diff(s)
    if abs(d0) < 1e-12:
        return None

    r = np.zeros(n)
    r[0] = 1.0
    for k in range(1, n):
        r[k] = -np.dot(dS[:k], r[k-1::-1]) / d0
    if not np.all(np.isfinite(r)):
        return None

    q = np.clip(r[::-1], 0.0, None)
    q /= q.sum()

    Mq = _M_dot(s, f, q)

    minimax, maximin = np.float32(Mq.max()), np.float32(Mq.min())
    saddle_gap = minimax - maximin
    if saddle_gap > MAX_SADDLE_GAP:
        return None
    else:
        value = 0.5 * (maximin + minimax)
        return SolveResult(
            minimax=minimax,
            maximin=maximin,
            saddle_gap=saddle_gap,
            value=value,
        )
        
def try_rung3(t: TransitionValues) -> SolveResult:
    s, f = np.asarray(t.success_values, float), float(t.fail_value)
    n = len(s)
    g = np.arange(n)[None, :] - np.arange(n)[:, None]
    M = np.where(g >= 0, s[g.clip(min=0)], f)

    c = np.zeros(n + 1); c[n] = -1.0                 # maximize v
    A_ub = np.hstack([-M.T, np.ones((n, 1))])        # v ≤ (pᵀM)_c  ∀c
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    bnds = [(0, None)] * n + [(None, None)]          # v free (the v=0 knife-edge)

    best = SolveResult()

    for method in ("highs-ds", "highs-ipm"):
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n),
                      A_eq=A_eq, b_eq=[1.0], bounds=bnds, method=method)
        if not res.success:
            continue
        p = np.clip(res.x[:n], 0, None); p /= p.sum()
        q = np.clip(-res.ineqlin.marginals, 0, None); q /= q.sum()
        maximin, minimax = float((p @ M).min()), float((M @ q).max())
        gap = minimax - maximin
        if gap <= MAX_SADDLE_GAP:
            return SolveResult(maximin, minimax, gap, 0.5*(maximin+minimax), 3)
        if gap < best.saddle_gap:
            best = SolveResult(maximin, minimax, gap, None, 3)
    return best                # value=None → caller aborts, per your spec

def solve(t: TransitionValues) -> SolveResult:
    res = try_rung1(t)
    if res.certified:
        return res
    else:
        res = try_rung2(t)
        if res.certified:
            return res
        else:
            res = try_rung3(t)
            if not res.certified:
                raise RuntimeError("uncertified class — aborting build")
            else:
                return res

    
        

            
                
                

                
        




def main():
    return None


if __name__ == "__main__":
    main()
