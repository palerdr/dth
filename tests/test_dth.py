"""Structural checks on the profile tables and the layers, the kernel against an independent numpy
certificate, and the dead-band sweep with its anchor."""
from types import SimpleNamespace
from decimal import Decimal, localcontext

import numpy as np
import pytest

import dth as m

N, ALIVE = m.N, m.N_ALIVE


def certificate(s, f):
    """Rungs 1-2 in numpy, independent of kernel.c: bounds (lower, upper) on the value of the stage
    matrix M[d, c] = s[c - d] for c >= d, else f."""
    lo, hi = max(s.min(), min(f, s[0])), min(s.max(), max(f, s[0]))
    if hi - lo <= m.MAX_SADDLE_GAP:
        return lo, hi
    d0, dS = s[0] - f, np.diff(s)
    r = np.zeros(60)
    r[0] = 1.0
    with np.errstate(all="ignore"):
        for k in range(1, 60):
            r[k] = -np.dot(dS[:k], r[k - 1::-1]) / d0
        r = np.clip(r, 0.0, None)
        r /= r.sum()
    g = np.arange(60)[None, :] - np.arange(60)[:, None]
    M = np.where(g >= 0, s[g.clip(min=0)], f)
    return (r @ M).min(), (M @ r[::-1]).max()


def class_values(pc, pd, V, T):
    s = -V[pd, T.succ[pc]]
    p = T.rev[pc]
    return s, p * (-V[pd, T.fail[pc]]) + (1.0 - p)


@pytest.fixture(scope="module")
def T():
    return m.build_table()


@pytest.fixture(scope="module")
def layer_sizes(T):
    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    return np.convolve(sizes, sizes)          # [P] = number of classes with potential P


# ---- tables --------------------------------------------------------------------------------------
def test_census(T):
    assert T.st.size == N == 17011
    assert (T.ttd[:ALIVE] != m.DEAD_PHI).all() and (T.ttd[ALIVE:] == m.DEAD_PHI).all()
    per_ttd = {t: int((T.ttd[:ALIVE] == t).sum()) for t in [0] + list(range(60, 301))}
    assert per_ttd[0] == 240
    assert all(per_ttd[t] == 241 - t for t in range(60, 241))
    assert all(per_ttd[t] == 0 for t in range(241, 301))
    order = np.lexsort((T.st[:ALIVE], T.ttd[:ALIVE]))   # normative order: TTD ascending, ST inside
    assert np.array_equal(order, np.arange(ALIVE))


def test_profile_ids(T):
    assert m.profile(0, 0, T) == 0 and T.alive_id[239, 0] == 239
    assert T.alive_id[0, 60] == 240 and T.alive_id[180, 60] == 420
    assert m.profile(240, 0, T) == 16951 and m.profile(299, 0, T) == 17010
    assert m.profile(240, 300, T) == 16951                  # dead: TTD discarded
    with pytest.raises(IndexError):
        m.profile(5, 30, T)                                 # alive TTD in 1..59 is off-domain


def test_phi_and_buckets(T):
    assert T.phi[:ALIVE].min() == 0 and T.phi[:ALIVE].max() == 240
    assert np.array_equal(T.phi[ALIVE:], T.st[ALIVE:] + 301)
    sizes = np.array([len(b) for b in T.bucket])
    assert sizes.size == 601 and sizes.sum() == N
    assert (sizes[241:301] == 0).all() and (sizes[301:] == 1).all()
    for v, b in enumerate(T.bucket):
        assert b.dtype == np.int32 and (T.phi[b] == v).all() and np.array_equal(b, np.sort(b))


def test_transitions(T):
    succ, lag = T.succ, np.arange(1, 61)
    assert np.array_equal(succ[0], np.arange(1, 61))            # (0,0): every lag survives
    assert np.array_equal(succ[239], np.arange(16951, 17011))   # (239,0): every lag lands dead
    assert (succ[17010] == m.WIN).all()                         # (299, dead): every lag hits capacity
    assert np.array_equal(succ == m.WIN, (T.st[:, None] + lag) >= 300)
    valid = succ != m.WIN
    assert (T.st[succ[valid]] == (T.st[:, None] + lag)[valid]).all()
    alive_child = succ < ALIVE
    assert (T.ttd[succ[alive_child]] == np.broadcast_to(T.ttd[:, None], succ.shape)[alive_child]).all()
    assert (succ[ALIVE:] >= ALIVE).all()                        # dead profiles have no alive children
    p = m.profile(0, 0, T)
    assert T.fail[p] == 240 and T.rev[p] == pytest.approx(0.95)
    p = m.profile(180, 60, T)
    assert T.fail[p] == ALIVE and T.rev[p] == pytest.approx(0.178125)   # t' = 300: dead sentinel at ST 0
    assert (T.fail[ALIVE:] == m.WIN).all() and (T.rev[ALIVE:] == 0.0).all()
    assert (T.rev[:ALIVE] > 0.0).all() and (T.rev[:ALIVE] <= 0.95).all()
    fc = T.fail[:ALIVE]
    revived = fc < ALIVE
    assert (T.st[fc[revived]] == 0).all()
    assert (T.ttd[fc[revived]] == (T.ttd[:ALIVE] + T.st[:ALIVE] + 60)[revived]).all()


def test_potential_increases(T):
    ok = T.succ != m.WIN
    assert (T.phi[T.succ[ok]] > np.broadcast_to(T.phi[:, None], T.succ.shape)[ok]).all()
    ok = T.fail != m.WIN
    assert (T.phi[T.fail[ok]] > T.phi[ok]).all()


def test_layer_census(layer_sizes):
    c = layer_sizes
    assert c.size == 1201 and c.sum() == 289_374_121
    assert c[:601].sum() == 283_297_201 and c[601:841].sum() == 6_015_600 and c[841:].sum() == 61_320
    assert c.argmax() == 374 and c[374] == 1_678_715


def test_layer_pairs(T, layer_sizes):
    for P in range(m.MAX_LAYER + 1):
        pcs, pds = m.layer_pairs(P, T)
        assert pcs.dtype == pds.dtype == np.int32 and pcs.size == layer_sizes[P]
    for P in (0, 374, 601, 1200):
        pcs, pds = m.layer_pairs(P, T)
        assert (T.phi[pcs] + T.phi[pds] == P).all()
        assert np.unique(pcs.astype(np.int64) * N + pds).size == pcs.size
    pcs, pds = m.layer_pairs(374, T)
    a0 = next(a for a in range(601) if T.bucket[a].size and T.bucket[374 - a].size)
    k = T.bucket[a0].size
    assert (pds[:k] == pds[0]).all() and np.array_equal(pcs[:k], T.bucket[a0])   # pd-major


# ---- the kernel ----------------------------------------------------------------------------------
def test_kernel_matches_certificate(T):
    """The kernel on 4,000 classes of layer 374 over a smooth synthetic value surface (both rungs and
    gate failures occur), against the numpy certificate. Only the rows the sample reads are filled."""
    rng = np.random.default_rng(0)
    pcs, pds = m.layer_pairs(374, T)
    pick = rng.choice(pcs.size, 4000, replace=False)
    pcs, pds = np.ascontiguousarray(pcs[pick]), np.ascontiguousarray(pds[pick])
    V = np.empty((N, N + 1))
    for pd in np.unique(pds):
        V[pd] = 0.9 * np.sin(np.linspace(0, 6, N + 1) + 3.0 * pd / N)
        V[pd, m.WIN] = -1.0
    V[pcs, pds] = np.nan                    # class cells have potential 374; no child cell does
    counts = np.zeros(2, dtype=np.int64)
    ptr = lambda a: a.ctypes.data  # noqa: E731
    failed = m.load_kernel().solve_chunk(pcs.size, ptr(pcs), ptr(pds), ptr(V), V.shape[1],
                                         ptr(T.succ), ptr(T.fail), ptr(T.rev), ptr(counts))
    assert counts[0] > 100 and counts[1] > 100 and failed > 100 and counts.sum() + failed == pcs.size
    edge = 0
    for i in range(pcs.size):
        pc, pd = int(pcs[i]), int(pds[i])
        lo, hi = certificate(*class_values(pc, pd, V, T))
        if np.isnan(V[pc, pd]):
            edge += hi - lo <= m.MAX_SADDLE_GAP     # summation order at the certificate edge
            continue
        assert hi - lo <= m.MAX_SADDLE_GAP and abs(V[pc, pd] - 0.5 * (lo + hi)) <= 1e-12
    assert edge <= 3


def test_dead_band_sweep(T, layer_sizes):
    STOP = 1082   # the dead-band anchor's layer; its whole subtree lives in layers >= 1082 (dead x dead)
    V, counts = m.build(T, stop_at=STOP)
    assert counts.sum() == layer_sizes[STOP:].sum()
    pcs, pds = np.nonzero(np.isfinite(V[:, :N]))
    assert pcs.size == counts.sum() and (T.phi[pcs] + T.phi[pds] >= STOP).all()
    vals = V[pcs, pds]
    assert (np.abs(vals) <= 1 + 1e-9).all()
    for i in range(0, pcs.size, max(1, pcs.size // 60)):
        lo, hi = certificate(*class_values(int(pcs[i]), int(pds[i]), V, T))
        assert hi - lo <= m.MAX_SADDLE_GAP and abs(vals[i] - 0.5 * (lo + hi)) <= m.MAX_SADDLE_GAP
    pc = pd = m.profile(240, 0, T)
    assert abs(V[pc, pd] - 0.3372132166291093) <= m.MAX_SADDLE_GAP


def synthetic_chunk(s, f):
    """We store synthetic child payoffs outside the output cells."""
    s = np.atleast_2d(s)
    n = s.shape[0]
    V = np.full((n, n + 61), np.nan)
    V[:, n:n + 60] = -s
    V[:, n + 60] = -f
    ids = np.arange(n, dtype=np.int32)
    succ = np.tile(np.arange(n, n + 60, dtype=np.int32), (n, 1))
    fail = np.full(n, n + 60, dtype=np.int32)
    rev = np.ones(n)
    counts = np.zeros(2, dtype=np.int64)
    ptr = lambda a: a.ctypes.data  # noqa: E731
    result = m.load_kernel().solve_chunk(n, ptr(ids), ptr(ids), ptr(V), V.shape[1],
                                        ptr(succ), ptr(fail), ptr(rev), ptr(counts))
    return result, counts, V[ids, ids]


@pytest.mark.parametrize("n", [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33])
def test_kernel_batch_tails(n):
    s = np.tile(np.linspace(-0.5, 0.5, 60), (n, 1))
    f = np.full(n, 0.8)
    s[::2] = 0.25
    f[::2] = 0.25
    result, counts, values = synthetic_chunk(s, f)
    assert result == 0 and counts.sum() == n
    for i, value in enumerate(values):
        lo, hi = certificate(s[i], f[i])
        assert hi - lo <= m.MAX_SADDLE_GAP
        assert abs(value - (lo + hi) / 2) <= 1e-12


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", [0, 1, 59, 60])
def test_kernel_rejects_invalid_children(bad, column):
    s, f = np.zeros(60), 0.0
    if column < 60:
        s[column] = bad
    else:
        f = bad
    result, counts, values = synthetic_chunk(s, f)
    assert result != 0 and counts.sum() == 0 and np.isnan(values[0])


def test_kernel_rejects_overflowed_equalizer():
    s = np.full(60, 0.5)
    # We choose finite weights whose sum exceeds float64 capacity.
    s[0] = -2.980687314913627e-6
    result, counts, values = synthetic_chunk(s, 0.0)
    assert result != 0 and counts.sum() == 0 and np.isnan(values[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.5])
def test_main_rejects_bad_anchor(T, monkeypatch, bad):
    V = np.empty((N, N + 1))
    V[0, 0] = bad
    monkeypatch.setattr(m, "build_table", lambda: T)
    monkeypatch.setattr(m, "build", lambda table: (V, np.array([m.CLASSES, 0])))
    monkeypatch.setattr(np, "save", lambda *args: pytest.fail("must reject the table before saving"))
    with pytest.raises(RuntimeError, match="anchor"):
        m.main()


def test_verify_rejects_overflowed_weights(monkeypatch):
    V = np.zeros((1, 62))
    V[0, 1:61] = -0.5
    V[0, 1] = 2.980687314913627e-6
    T = SimpleNamespace(succ=np.arange(1, 61)[None, :], fail=np.array([61]), rev=np.ones(1))
    monkeypatch.setattr(m, "ANCHORS", [])
    monkeypatch.setattr(m, "CLASSES", 1)
    with pytest.raises(RuntimeError, match="weight sum"):
        m.verify(V, T)


@pytest.mark.parametrize("s,t", [(-1, 0), (300, 0), (0, -1), (0, 301)])
def test_profile_rejects_out_of_range(T, s, t):
    with pytest.raises(IndexError):
        m.profile(s, t, T)


@pytest.mark.parametrize("options", [{"stop_at": -1}, {"stop_at": 1201}, {"threads": 0}])
def test_build_rejects_invalid_options(T, options):
    with pytest.raises(ValueError):
        m.build(T, **options)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("case", range(10))
def test_nonnegative_certificate_against_decimal(sign, case):
    if case < 6:
        d, ratio = [(0.25, 0.01), (0.25, 0.9), (0.25, 1.0), (0.25, 2.0),
                    (2e-6, 165000.0), (1.01e-6, 1.1)][case]
        f = 0.0
        s = np.full(60, d * (ratio - 1))
        s[0] = -d
    else:
        a = [0.1, 0.9, 1.0, 1.2][case - 6]
        f = 0.25
        s = np.full(60, 0.05 + 0.4 * a - 0.2 * a * a)
        s[:2] = [0.05, 0.05 + 0.4 * a]
    s, f = sign * s, sign * f
    pure_lo = max(s.min(), min(f, s[0]))
    pure_hi = min(s.max(), max(f, s[0]))
    assert pure_hi - pure_lo > m.MAX_SADDLE_GAP
    result, counts, values = synthetic_chunk(s, f)
    assert result == 0 and counts.tolist() == [0, 1]
    with localcontext(prec=90):
        S = [Decimal.from_float(float(x)) for x in s]
        F = Decimal.from_float(float(f))
        d = S[0] - F
        delta = [S[k] - S[k - 1] for k in range(1, 60)]
        r = [Decimal(1)]
        for k in range(1, 60):
            r.append(-sum(delta[j] * r[k - 1 - j] for j in range(k)) / d)
        assert min(r) >= 0
        exact = F + d / sum(r)
        assert abs(Decimal.from_float(float(values[0])) - exact) < Decimal("1e-10")


def test_clipped_weights_use_full_certificate():
    s = np.full((19, 60), 0.5)
    s[:, 0] = 0.0
    s[:, -1] = -1e-7
    s[1::2] = np.linspace(-0.5, 0.5, 60)
    result, counts, values = synthetic_chunk(s, np.full(19, 0.5))
    assert result == 0 and counts.tolist() == [0, 19]
    for i, value in enumerate(values):
        lo, hi = certificate(s[i], 0.5)
        assert hi - lo <= m.MAX_SADDLE_GAP and abs(value - (lo + hi) / 2) < 1e-12
    # We distinguish the clipped certificate from the invalid weight-sum shortcut.
    assert abs(values[0] - (0.5 - 0.5 / 59)) > 1e-10
