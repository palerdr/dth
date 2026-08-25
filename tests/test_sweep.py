"""Step 5: the layer sweep on the dead band, checkpoint/resume, recheck."""
import shutil

import numpy as np
import pytest

import main as m

DEAD_ANCHOR = ((240, 0, 240, 0), 0.3372132166291093)
STOP = 1082   # the dead-band anchor's layer; its whole subtree lives in layers >= 1082 (dead x dead only)


@pytest.fixture(scope="module")
def T():
    return m.build_table()


@pytest.fixture(scope="module")
def counts(T):
    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    return np.convolve(sizes, sizes)


def quiet(_msg):
    pass


def test_dead_band_anchor(T, counts):
    V, K, ck = m.sweep("unused", stop_at=STOP, save=False, table=T, log=quiet)
    assert ck.next == STOP - 1 and not ck.complete
    pc = pd = m.profile(240, 0, T)
    assert abs(V[pc, pd] - DEAD_ANCHOR[1]) <= m.MAX_SADDLE_GAP
    solved = ~np.isnan(V[:, :m.N])
    assert np.array_equal(solved, K != m.UNSOLVED)
    assert solved.sum() == counts[STOP:].sum() == sum(ck.counts)
    pcs, pds = np.nonzero(solved)
    assert (T.phi[pcs] + T.phi[pds] >= STOP).all()
    assert (np.abs(V[pcs, pds]) <= 1 + 1e-9).all()
    # every solved class re-derives independently to within the gate
    for i in range(0, pcs.size, max(1, pcs.size // 60)):
        c = int(pcs[i]) * m.N + int(pds[i])
        assert abs(m.recheck(c, V, T).value - V[pcs[i], pds[i]]) <= m.MAX_SADDLE_GAP
    ok = {state: good for state, _e, _g, good in m.verify_anchors(V, T)}
    assert ok[DEAD_ANCHOR[0]]


def test_checkpoint_resume_is_idempotent(tmp_path, T):
    d = str(tmp_path / "build")
    try:
        V1, K1, ck1 = m.sweep("unused", stop_at=STOP, save=False, table=T, log=quiet)
        m.sweep(d, max_layers=40, stop_at=STOP, checkpoint_minutes=1e9, table=T, log=quiet)
        _V, _K, ck = m.load_build(d)
        assert ck.next == m.MAX_LAYER - 40
        V2, K2, ck2 = m.sweep(d, stop_at=STOP, checkpoint_minutes=1e9, table=T, log=quiet)
        assert ck2.next == ck1.next and ck2.counts == ck1.counts
        assert np.array_equal(V1, V2, equal_nan=True) and np.array_equal(K1, K2)
    finally:
        shutil.rmtree(d, ignore_errors=True)
