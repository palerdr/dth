"""Step 2: value storage, layer enumeration, batched gather."""
import time

import numpy as np
import pytest

import main as m

N = m.N


@pytest.fixture(scope="module")
def T():
    return m.build_table()


@pytest.fixture(scope="module")
def V():
    rng = np.random.default_rng(0)
    V = rng.uniform(-1.0, 1.0, size=(N, N + 1))
    V[:, m.WIN] = -1.0
    return V


@pytest.fixture(scope="module")
def counts(T):
    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    return np.convolve(sizes, sizes)


def test_storage_layout():
    V = m.values_array()
    assert V.shape == (N, N + 1) and V.dtype == np.float64 and V.flags.c_contiguous
    assert np.isnan(V[:, :N]).all() and (V[:, m.WIN] == -1.0).all()
    K = m.kinds_array()
    assert K.shape == (N, N) and K.dtype == np.uint8 and (K == m.UNSOLVED).all()


def test_gather_matches_scalar_oracle(T, V):
    rng = np.random.default_rng(1)
    pcs = rng.integers(0, N, 2000).astype(np.int32)
    pds = rng.integers(0, N, 2000).astype(np.int32)
    S, F = m.gather(pcs, pds, V, T)
    assert S.shape == (2000, 60) and F.shape == (2000,) and S.dtype == np.float64
    for i in range(2000):
        s, f = m.class_values(int(pcs[i]), int(pds[i]), V, T)
        assert np.array_equal(S[i], s)
        assert F[i] == f


def test_gather_win_and_dead_cases(T, V):
    pd = 1234
    p180 = m.profile(180, 60, T)
    pcs = np.array([17010, 239, 16711, p180], dtype=np.int32)
    S, F = m.gather(pcs, np.full(4, pd, dtype=np.int32), V, T)
    assert (S[0] == 1.0).all() and F[0] == 1.0                     # (299, dead): every child is WIN
    assert np.array_equal(S[1], -V[pd, 16951:17011])                 # (239, 0): the dead run
    assert F[2] == 1.0                                               # a dead Checker loses every failed check
    rev = T.rev[p180]
    assert F[3] == rev * (-V[pd, m.N_ALIVE]) + (1.0 - rev)          # revived at ST 0 but cannot survive


def test_layer_pairs(T, counts):
    total = 0
    for P in range(m.MAX_LAYER + 1):
        pcs, pds = m.layer_pairs(P, T)
        assert pcs.dtype == np.int32 and pds.dtype == np.int32
        assert pcs.size == counts[P]
        if pcs.size:
            assert (T.phi[pcs] + T.phi[pds] == P).all()
            keys = pcs.astype(np.int64) * N + pds
            assert np.unique(keys).size == keys.size
        total += pcs.size
    assert total == m.CLASSES
    # pd-major inside the first rectangle of the largest layer
    pcs, pds = m.layer_pairs(374, T)
    a0 = max(0, 374 - m.MAX_PHI)
    while T.bucket[a0].size == 0 or T.bucket[374 - a0].size == 0:
        a0 += 1
    k = T.bucket[a0].size
    assert (pds[:k] == pds[0]).all() and np.array_equal(pcs[:k], T.bucket[a0])


def test_gather_detects_unsolved_child(T, V):
    pc, pd = 5, 77
    child = int(T.success_children[pc, 3])
    saved = V[pd, child]
    V[pd, child] = np.nan
    try:
        with pytest.raises(RuntimeError):
            m.gather(np.array([pc], np.int32), np.array([pd], np.int32), V, T)
    finally:
        V[pd, child] = saved


def test_chunks_cover_layer(T, V):
    P = 374
    pcs, pds = m.layer_pairs(P, T)
    t0 = time.perf_counter()
    parts = [m.gather(pc, pd, V, T) for _lo, _hi, pc, pd in m.iter_chunks(P, T)]
    dt = time.perf_counter() - t0
    S = np.concatenate([p[0] for p in parts])
    F = np.concatenate([p[1] for p in parts])
    assert S.shape[0] == pcs.size == 1_678_715
    idx = np.random.default_rng(2).integers(0, pcs.size, 500)
    S2, F2 = m.gather(pcs[idx], pds[idx], V, T)
    assert np.array_equal(S[idx], S2) and np.array_equal(F[idx], F2)
    print(f"\nfull-layer gather P=374: {dt:.2f}s ({pcs.size / dt:.0f} classes/s)")
