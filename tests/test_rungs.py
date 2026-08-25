"""Steps 3-4: batched rungs against the scalar oracles and against LP-certified values."""
import numpy as np
import pytest

import main as m


def random_batch(rng, B):
    return rng.uniform(-1, 1, (B, 60)), rng.uniform(-1, 1, B)


def structured_batch(rng, B):
    # success values monotone in the lag, as real classes tend to be
    return np.sort(rng.uniform(-1, 1, (B, 60)), axis=1), rng.uniform(-1, 1, B)


def test_solve_result_post_init():
    r = m.SolveResult(maximin=0.25, minimax=0.25 + 5e-7, kind=m.SUPPORT)
    assert r.certified and r.saddle_gap == pytest.approx(5e-7) and r.value == pytest.approx(0.25 + 2.5e-7)
    assert r.kind == m.SUPPORT
    r = m.SolveResult(maximin=0.0, minimax=1e-3, kind=m.LP)
    assert not r.certified and r.value is None
    r = m.SolveResult(maximin=np.nan, minimax=np.nan, kind=m.LP)
    assert not r.certified and r.value is None
    r = m.SolveResult(maximin=0.3, minimax=0.3 - 1e-15, kind=m.PURE)   # rounding can make the gap slightly negative
    assert r.certified


def test_toeplitz_matvec_matches_scalar():
    rng = np.random.default_rng(3)
    S, F = random_batch(rng, 64)
    q = rng.uniform(0, 1, (64, 60))
    q /= q.sum(1, keepdims=True)
    Mq = m.toeplitz_matvec(S, F, q)
    for i in range(64):
        assert np.allclose(Mq[i], m._M_dot(S[i], F[i], q[i]), atol=1e-12, rtol=0)
        assert np.allclose(Mq[i], m.full_matrix(S[i], F[i]) @ q[i], atol=1e-12, rtol=0)


def test_rung1_batch_matches_scalar():
    rng = np.random.default_rng(4)
    S, F = random_batch(rng, 500)
    lo, hi = m.rung1_batch(S, F)
    for i in range(500):
        r = m.try_rung1(S[i], F[i])
        assert lo[i] == r.maximin and hi[i] == r.minimax


def test_rung2_batch_matches_scalar():
    rng = np.random.default_rng(5)
    for maker in (random_batch, structured_batch):
        S, F = maker(rng, 300)
        lo, hi = m.rung2_batch(S, F)
        mismatched_usability = 0
        for i in range(300):
            r = m.try_rung2(S[i], F[i])
            if (r is None) != bool(np.isnan(lo[i])):
                mismatched_usability += 1      # rounding at the overflow edge can differ; must be rare
                continue
            if r is not None:
                assert lo[i] == pytest.approx(r.maximin, abs=1e-9)
                assert hi[i] == pytest.approx(r.minimax, abs=1e-9)
        assert mismatched_usability <= 2


def test_bounds_bracket_lp_value():
    rng = np.random.default_rng(6)
    S, F = structured_batch(rng, 60)
    lo1, hi1 = m.rung1_batch(S, F)
    lo2, hi2 = m.rung2_batch(S, F)
    for i in range(60):
        res = m.try_rung3(S[i], F[i])
        assert res.certified and res.kind == m.LP
        v = res.value
        assert lo1[i] <= v + 2e-6 and hi1[i] >= v - 2e-6
        if not np.isnan(lo2[i]):
            assert lo2[i] <= v + 2e-6 and hi2[i] >= v - 2e-6


def test_solve_chunk_matches_solve_class():
    rng = np.random.default_rng(7)
    S, F = structured_batch(rng, 200)
    S[0] = 0.5; F[0] = 0.5                          # everything equal: pure, value 0.5
    S[1] = 1.0; F[1] = 1.0                          # every continuation is a win
    S[2] = np.linspace(-1, 1, 60); F[2] = -1.0      # the Checker's c = 1 pins the Dropper at -1
    vals, kinds = m.solve_chunk(S, F)
    assert kinds[0] == m.PURE and vals[0] == 0.5
    assert kinds[1] == m.PURE and vals[1] == 1.0
    assert kinds[2] == m.PURE and vals[2] == -1.0
    assert vals.dtype == np.float64 and kinds.dtype == np.uint8 and (kinds <= m.LP).all()
    for i in range(200):
        r = m.solve_class(S[i], F[i])
        assert abs(vals[i] - r.value) <= m.MAX_SADDLE_GAP + 1e-12   # two certified midpoints differ by at most the gate
    print(f"\nroute mix on the structured batch (pure, support, lp): {np.bincount(kinds, minlength=3)}")
