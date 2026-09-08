# You build the Rust solver

You will port your completed [stl.ipynb](../stl.ipynb) exercises into a native
tablebase solver. You will retain the current algorithm, including its 16-lane
recurrence and weight-sum certificate. You will measure Rust performance after
you pass the value checks.

You have a Rust 2024 crate with profile rules in `profiles` and layer scheduling
in `sweep`. You use `thiserror` for the error type and `cargo test --locked` to
check these pieces.

You still need to implement the stage solver and full tablebase sweep. The
starter command exits with status 2 and directs you to this guide.

From the repository root, enter `solver`. Run the commands in this guide from
that crate directory unless a block states another working directory:

```sh
cd solver
cargo build
cargo fmt --check
```

You will find contracts and acceptance values below. You choose the function
bodies. You can consult your notebook before you read the existing C code.

## Your build sequence

| Milestone | Notebook foundation | Result you can check |
| --- | --- | --- |
| 1. Model and profile IDs | E01–E06 | Stable IDs and exact transition tables |
| 2. Scalar stage solver | E08–E17 | Small matrices with full certificates |
| 3. Sum certificate | E18–E20 | The fast path with its proof conditions |
| 4. Endgame and table layout | E21 | A 16-state solve and child addressing |
| 5. Full sweep | E07, E22, E24 | A complete table with one worker |
| 6. Batched kernel | E23 | Sixteen independent classes per batch |
| 7. Worker pool | E24 | Parallel reads with a layer barrier |
| 8. Verification and output | E09, E22 | A checked NumPy-compatible table |
| 9. Performance | E17, E23 | Timings for the complete command |

You can start in `src/lib.rs` and keep command-line parsing in `src/main.rs`.
If you need module boundaries, use these responsibilities:

| Possible module | Responsibility |
| --- | --- |
| `profiles` | Rules, canonical IDs, and precomputed transitions |
| `stage` | Scalar reference and batched stage calculation |
| `sweep` | Layer schedule, result buffers, and worker pool |
| `main` | Options, verification, and output |

You need no trait hierarchy or generic game engine. You can use fixed-size
arrays for the 60-action kernel and keep small dynamic matrices in tests.

You use `n` for actions and `N` for profiles. In the notebook's stage helpers,
`n = len(S)`; a separate `n` parameter would duplicate that information.
Your production Rust input can be `&[f64; 60]`, which fixes `n = 60` in its
type. A test helper can accept a nonempty slice and derive `n = S.len()`.
Keep the sum-certificate route restricted to 60 actions.

You follow these shared boundary contracts:

| Operation | Input and result | Error behavior |
| --- | --- | --- |
| Profile reduction | Raw integer ST/TTD to a canonical profile or ID | Reject invalid ranges and TTD 1 through 59 before reduction |
| Stage reference | Nonempty `S`, scalar `F`, and `n = S.len()` | Reject nonfinite payoffs before choosing a route |
| Probability pair | Two length-`n` vectors, with finite nonnegative entries | Reject a sum outside absolute tolerance `1e-12` of one |
| Stage result | Scalar value and a route | Reject nonfinite values or `abs(value) > 1 + 1e-9` on any route |
| Layer generation | A layer number and profile buckets | Return ordered `(checker_id, dropper_id)` pairs; empty layers have no pairs |
| Layer commit | Read-only table, unique output cells, separate result buffer | Commit after all evaluations and result checks succeed |

You port the notebook's invalid-input checks with its expected values.
A passing fixture checks one behavior. You need the stated proof and its
preconditions before treating an acceptance route as a certificate.

## 1. Model and profile IDs

You store the value from the current Dropper's perspective. Your state is the
ordered pair `(checker_id, dropper_id)`. After a live move, you swap the roles
and negate the child's value. You must preserve the distinction between the
state's first index and the player whose payoff you store.

You use these frozen model constants:

| Quantity | Value |
| --- | --- |
| Actions and success lags | 60; lag is `c - d + 1` for game seconds with `c >= d` |
| ST domain and capacity | `0..=299`; a success reaching 300 gives the Dropper `+1` |
| TTD domain, for cumulative time dead | `0` or `60..=300` before reduction |
| Failure dose | `s + 60` |
| Eligibility | `s <= 239 && s + t <= 240` |
| Revival probability | $0.95(1-s/240)0.75^{t/60}$ on eligible profiles; zero outside that region |
| Failure-fatal potential offset | 301 |

You perform the probability divisions in `f64` and use its power operation.
The exponent in the table is mathematical notation; Rust's `^` operator means
bitwise XOR. Integer division would change the model. You expect small
cross-language differences from the power function; compare probabilities
and values with tolerances.

You can begin with an enum for an eligible profile and a failure-fatal profile.
You can flatten it into arrays after you check the census. A failure-fatal player
still takes turns and can win. You discard that player's TTD because success
cannot restore eligibility and failure ends the game.

You assign IDs by ascending TTD, then ascending ST, for eligible profiles.
You append the 300 failure-fatal profiles in ascending ST order. You obtain
16,711 eligible profiles and 17,011 total profiles. You reject any input with
TTD in `1..=59`, including failure-fatal profiles, before reduction. You reject
out-of-range ST or TTD at this same boundary.

You use the closed TTD domain from the notebook. In the existing Python lookup,
we also accept failure-fatal inputs with TTD in `1..=59`. You preserve its IDs
and transitions for valid-domain inputs with the notebook's validation rule.

You precompute the following data once:

| Data | Suggested representation | Invariant |
| --- | --- | --- |
| Success child | `Vec<[u32; 60]>` | Child profile ID or terminal column ID |
| Revival child | `Vec<u32>` | Reduced child at ST zero, or terminal ID when revival is absent |
| Revival probability | `Vec<f64>` | Finite and in `[0, 1]`; zero for failure-fatal profiles |
| Potential | `Vec<u16>` | Eligible `s + t`; failure-fatal `s + 301` |
| Buckets | 601 lists of IDs | Ascending IDs within each potential |

You update a successful profile to ST `s + lag`, with a terminal at capacity.
For a live child of a failure-fatal profile, keep the failure-fatal form and the
new ST. The notebook's `None` inside a profile corresponds to that form; a
whole child of `None` denotes a terminal or absent transition. Keep those two
cases distinct in your enum and transition representation.
You update a revived profile to `(0, t + s + 60)` and reduce it again. A player
can survive this round and become failure-fatal for the next one.

You use an integer type wide enough for intermediate sums. A raw `s + t`
can reach 599, and `s + lag` can reach 359. A `u16` can represent both.
Validate signed command inputs before converting them to unsigned IDs.

**You prove:** both updates stay in the reduced domain. For a live child, you
also prove a strict potential increase. E05 has the four cases; test this
property over the complete profile transition tables before a sweep.

**You check these fixtures:**

| Input or transition | Expected result |
| --- | --- |
| ID of `(0, 0)` | 0 |
| ID of `(239, 0)` | 239 |
| ID of `(0, 60)` | 240 |
| ID of `(180, 60)` | 420 |
| ID of `(0, 240)` | 16,710 |
| Failure-fatal profile at ST zero | 16,711 |
| ID of `(240, 0)` or `(240, 300)` | 16,951 |
| Failure-fatal profile at ST 299 | 17,010 |
| Success from ID 0 with lag 1 | ID 1 |
| Success from ID 239 with lag 1 | ID 16,951 |
| Revival from ID 0 | ID 240, with probability 0.95 |
| Revival from ID 420 | ID 16,711, with probability 0.178125 |
| Success from failure-fatal ST 250 with lag 1 | Failure-fatal ST 251 |
| Success from failure-fatal ST 240 with lag 60 | Terminal column ID 17,011 |
| Revival probability at `(181, 60)` or `(250, 0)` | Zero |
| Raw input `(299, 1)` or `(0, 301)` | Error |

You can move on after the census and transition invariants pass in debug and
release tests. Your tests should use `assert!`; do not make correctness checks
depend on `debug_assert!` in the production solver.

## 2. Scalar stage solver

You give the stage function 60 success payoffs and one failure payoff. You return
a value and its acceptance route, or an error. You can retain lower and upper
bounds in a test result to inspect failures. You need no per-state heap allocation
in the finished stage calculation.

You use zero-based action indices in Rust:

$$
M_{d,c}=\begin{cases}S_{c-d}&c\ge d,\\F&c<d.\end{cases}
$$

You index rows by Dropper actions and columns by Checker actions. You place
`S[0]` on the diagonal because equal seconds give inclusive lag one. For a
length-one test input, the matrix is `[[S[0]]]`. You validate `F` even though
that one-action game has no failure entry.

You first port the pure bounds and the recurrence with a full certificate:

$$
L_0=\max(\min S,\min(F,S_0)),\qquad
U_0=\min(\max S,\max(F,S_0)).
$$

$$
d_0=S_0-F,\quad b_m=(S_{m-1}-S_m)/d_0,\quad
r_0=1,\quad r_k=\sum_{j<k}b_{k-j}r_j.
$$

You clip finite negative weights to zero and normalize for the candidate pair.
The Dropper uses the weights in forward order; the Checker uses reverse order.
You calculate row payoffs with

$$
x_i=F\sum_{c<i}q_c+\sum_{j=0}^{59-i}S_jq_{i+j}.
$$

You use `min(x)` and `max(x)` as the pair's bounds. You accept their midpoint
at a gap of at most `1e-6`. You can use an explicit matrix and both products in
tests to check the structured product and the reversal.

**You prove:** subtracting adjacent rows gives the recurrence. Reversal gives
`J M J = transpose(M)`, so one product bounds both players. After clipping, the
pair remains legal, and the full certificate determines acceptance. You make no
claim that clipping preserves the equalizer equations.

You check a pure result before dividing by `d0`. For an unresolved mixed stage,
you require `abs(d0) >= 1e-12`. You reject nonfinite raw weights and a nonfinite
or nonpositive clipped sum. You check the sum after accumulation, since finite
weights can overflow it. You reject a nonfinite or out-of-range final value.

You can give a stage error its input class when the sweep calls it. Include
the layer and profile IDs in the error report. You stop before saving a table
if any class fails certification.

You should port the notebook checks, including the four-action geometric
fixture with weights `[1, 2, 4, 8]` and value `-1/60`. You can keep a small
slice-based test helper while the production entry point uses 60-element arrays.
You should also test a non-pure input that fails the certificate gate.

You check shape and orientation with the one-action case and an asymmetric
matrix. You test zero probabilities as legal inputs. For `S=[0,0,1]` and
`F=1`, you obtain `r=[1,0,1]`, value `1/2`, and a strategy pair with
partial support. The absence of an LP fallback does not establish full support.

You retain a reference recurrence that divides after its dot product. Your
optimized recurrence divides the coefficients before the dot products. Compare
those two paths with the dense matrix test, so a shared coefficient-index error
cannot pass through a comparison of two copies of the same code.

## 3. Sum certificate

You can remove the second triangular product from the common mixed path. For
the raw recurrence weights, let `W` be their exact sum. If those weights are
nonnegative, the last row's payoff is `F + (S[0] - F) / W`. You use the residual
proof to bound the effect of the floating recurrence.

You keep these conditions together at the routing boundary:

| Condition | Reason |
| --- | --- |
| 60 actions and ordinary binary64 arithmetic | You use the constants proved for this operation count |
| Round-to-nearest with gradual underflow; no fast-math | You preserve the roundoff model |
| Finite payoffs in `[-2, 2]` | You bound coefficient numerators and payoff errors |
| `abs(d0) >= 1e-6` | You stay inside the shortcut's stated domain |
| Raw weights from this stage's coefficient recurrence, with `r[0] = 1` | You preserve the residual identity and its provenance |
| Finite nonnegative raw weights | You bound subset mass by the total mass |
| Finite positive sum, accumulated in reverse index order | You can compute a finite candidate with the stated summation bound |

You require nonnegative weights. Zero weights need no clipping and can qualify.
You cannot accept arbitrary positive weights through this path. Keep recurrence
formation and shortcut selection inside the same stage or batch function.

**You prove:** for the computed weights, the normalized row-equation residual is
less than `6e-14`. You use the exact sum of the computed weights to define
an exact probability pair in this proof; you need no rounded normalization on
the sum route. Across 59 adjacent differences, you bound the row-payoff spread
by `3.54e-12`. You add less than `4e-14` for the stored last-row formula and use
an interval radius of `1e-10`. You can read the full derivation in
[E18–E19](../stl.ipynb) and [the certificate proof](../src/certificate.md).

You fall back to the full matrix certificate after clipping or outside the sum
certificate's domain. If the full certificate fails, you return an error. You
need no linear-program dependency to reproduce the fixed model's current solve.

You preserve the proof's arithmetic contract in your standalone process. Rust's
`f64` supplies binary64 values; you still need to document the rounding and
underflow environment you support. A runtime underflow smoke test can check that
half the smallest normal value remains nonzero. That check alone does not prove
the rounding mode. If you later embed the kernel in a host that can change the
floating environment, add a platform check or make the default environment an
explicit entry precondition. The existing C kernel checks both properties.
You must not reuse its environment guarantee without implementing that check.
For a safe Rust implementation with `forbid(unsafe_code)`, you can state the
default arithmetic environment as a precondition for each worker. If you add a
platform environment query, choose a safe API with a documented contract.
A smoke test alone cannot justify a claim that you reject changed rounding modes.

You can permit `mul_add` under the conservative residual bound. Measure its
performance and compare results before you choose it. You should consult the
[Rust `f64` API](https://doc.rust-lang.org/std/primitive.f64.html) for arithmetic
and finite-value operations. You keep architecture-specific fast-math flags out
of your build.

You need these numerical tests in addition to the notebook's rational fixtures:

| Fixture | Expected behavior |
| --- | --- |
| NaN or infinity in a success or failure payoff | Reject before pure routing |
| Finite weights whose sum overflows | Reject before normalization |
| `S[0]=0`, indices 1 through 58 equal 0.5, `S[59]=-1e-7`, `F=0.5` | Use the full certificate after clipping |
| The same clipped shape with four actions and last success `-0.1` | Reject the wide gap |
| Signed coefficients with nonnegative weights | Permit the sum certificate if its other conditions hold |
| Small accepted value in the subnormal range | Preserve the value under the stated arithmetic environment |

In the 60-action clipped fixture, compare the accepted midpoint with
`0.5 - 0.5 / 59`; their difference exceeds `1e-10`. This fixture catches accidental use of the sum shortcut after
clipping.

You can create signed-coefficient fixtures with `F=0.25`, `S[0]=0.05`,
`S[1]=0.05+0.4*a`, and later successes `0.05+0.4*a-0.2*a*a`. In exact arithmetic
the recurrence has coefficients `b1=2*a`, `b2=-a*a` and weights
`r[k]=(k+1)*a^k`. You can choose `a=0.9` or `a=1.2` to test positive weights
despite a negative coefficient.

You can obtain the overflow and high-precision stress fixtures from
[the current tests](../tests/test_dth.py). You should compare against their
90-digit Decimal calculations outside the timed Rust build.

## 4. Endgame and table layout

You first port E21's closed endgame at failure-fatal ST 296 through 299.
You use local IDs `0..=3` in ST order for its 4-by-4 table. Local ID zero means
ST 296; these local IDs differ from the full catalog's IDs.
You keep all 60 actions and solve 16 ordered states. You check the complete
result against your notebook, including `V[299, s] = 1` and
`V[298, 299] = 29/30`, where these labels denote ST values.

For the full table, you allocate one row-major `Vec<f64>` with
`N * (N + 1)` elements. Set the class cells to NaN and the last column to `-1`.
You designate `WIN = N = 17011` as that column's ID. This is a column sentinel;
it is not an extra profile row.

You gather from the row of the current Dropper:

$$
S_k=-V[p_d,\operatorname{succ}(p_c,k)],\qquad
F=a(p_c)(-V[p_d,\operatorname{fail}(p_c)])+(1-a(p_c)).
$$

You write the result at `V[pc, pd]`. For an ordinary live child, the updated
Checker becomes the child's Dropper, which puts its ID in the child column.
For a terminal success, negating the sentinel value gives `+1`. For absent
revival, probability zero and a finite sentinel read give failure payoff `+1`.

**You prove:** row-major offset `row * (N + 1) + column` represents the role
swap from E01. Check it with a table whose cells have distinct synthetic values.
You should test the gathered payoffs before you test the resulting game value.

You use `usize` for address arithmetic and allocation sizes, with checked size
multiplication. You can retain `u32` transition IDs and convert them for indexing.
The values occupy 2,315,129,056 bytes, excluding the NumPy header and work buffers.
You keep a full allocation on the heap and small fixed scratch arrays per task.

You do not halve this allocation through player antisymmetry. The root has equal
profile IDs and a nonzero value; zero-sum payoffs do not imply antisymmetry under
an exchange of the decision roles.

## 5. Full sweep with one worker

You enumerate layer `P` from bucket products `B[a] × B[P-a]`. Visit `a` in
ascending order. Within a product, hold one Dropper ID fixed while visiting
Checker IDs in ascending ID order. Advance the Dropper ID after that group.
You group adjacent tasks that read the same child row. You allocate pairs for
one layer and reuse the capacity on the next layer.

You check order with the small catalog
`[(1,60), (0,60), (61,0), (60,0)]`. Its local IDs are 0 through 3 and
its potentials are `[61,60,61,60]`. At layer 121, require
`[(1,0),(3,0),(1,2),(3,2),(0,1),(2,1),(0,3),(2,3)]`.
Each bucket has two IDs, so the fixture detects exchanged inner loops.
The catalog starts with potential 61, so it detects reliance on insertion order.

You check empty catalogs and missing buckets before the full sweep. For an
integer layer outside `0..=1200`, a general pair helper returns no pairs.
Your command parser rejects a `stop_at` outside that range.

You can count layers from the bucket histogram before you allocate the table:

| Check | Expected result |
| --- | --- |
| Number of layer counts | 1201 |
| Sum of layer counts | 289,374,121 |
| Largest layer | 374 |
| Classes in that layer | 1,678,715 |

You compute a layer into a temporary result vector while you borrow the table
for reads. After a complete successful calculation, you commit the layer.
With a `Vec<f64>` for results, the largest layer needs 13,429,720 bytes.
The pair buffers and worker scratch require additional memory. You can
keep route counts per chunk instead of storing a padded result struct per class.

**You prove:** the profile transition checks imply that a class reads children
from completed layers. A role swap preserves the potential sum. Each bucket
product generates distinct output cells, and each class appears once. You can
defer writes without changing the calculation.

You support an internal `stop_at` for partial sweeps, in `0..=1200`. A partial
sweep leaves lower layers NaN. You distinguish it from a complete build in its
report, and you withhold the final table file until the full verification passes.

You can test the failure-fatal band down to layer 1082 before a full build.
The state `(240, 0, 240, 0)` lies there after profile reduction. Its reference
value is `0.3372132166291093` within the `1e-6` comparison tolerance.

You then run the full scalar sweep once and keep its output for comparison.
You expect it to take longer than the final batched build. You need this scalar
reference to separate indexing errors from later layout changes.

## 6. Batched kernel

You extend E23 to fixed arrays with type shape `[[f64; 16]; 60]`. You use
action-major storage so the lane index is contiguous. At fixed recurrence
indices `k,j`, you update 16 independent accumulators.

You keep the current pull-loop organization: choose output `k`, zero the lane
accumulators, visit earlier weights `j`, and store that output after the dot
products finish. You form coefficients before the recurrence. You keep the
summation order within a lane when you interleave lanes.

You can reuse the same storage through the phases:

| Phase | Success array | Weight array | Third work array |
| --- | --- | --- | --- |
| Gather | Success payoffs | Unused | Unused |
| Coefficients | Retained | Unused | Coefficients `b` |
| Recurrence | Retained | Raw `r` | Coefficients `b` |
| Routing | Retained | Raw `r` | Reversed clipped weights, normalized if needed |

You need 23,040 bytes for these arrays, plus lane scalars and flags. You can
retain `F` and the pure bounds while gathering. You scan raw weights for finite
values and negativity during the later sum phase. You need the full product
for the rare mixed lanes outside the sum certificate's contract.

**You prove:** one lane follows the scalar recurrence. Scratch values from a
different lane cannot affect it. Reusing coefficient storage after the recurrence
is safe because the remaining phases no longer read those coefficients.

You can return before the batch loop for zero input classes. For a short final
batch, fill spare lanes from a valid input lane and remember the actual length.
You write and count that length. You must not read `pairs[base]` for an empty batch.

You can leave pure lanes in the recurrence work to keep the inner loops uniform.
For a pure lane with zero `d0`, choose a harmless denominator for unused scratch
work. Route that lane from its pure bounds. You must still reject nonfinite
gathered inputs before routing, and you must not apply that denominator substitution
to an unresolved mixed lane.

You can first test lengths `0, 1, 15, 16, 17, 31, 32, 33`, with alternating
pure and mixed fixtures. Check output sentinels beyond the valid length. Add
mixed batches with both sum and clipped routes. You use the scalar
full-certificate reference for the value comparison.

You can inspect optimized assembly after these tests pass. Lane count 16 is a
software batch size. On Apple NEON, one vector holds two `f64` values. A source
loop over 16 lanes does not imply a 16-wide hardware instruction.

## 7. Worker pool and Rust borrowing

You keep the library's `forbid(unsafe_code)` while you implement the read/commit
design. During a layer calculation, workers share the immutable table and
transition arrays. Each worker owns scratch and a disjoint part of the output
buffer. You end those borrows before you commit results through a mutable table
borrow.

**You prove two properties:** the game graph permits independent work within
the layer, and Rust's references represent disjoint writes to the result buffer.
The graph proof alone does not permit `&[f64]` and `&mut [f64]` over the same
table allocation. An in-place raw-pointer port needs its own unsafe abstraction
and aliasing proof. You can measure the safe staging cost before considering it.

You can add Rayon after the scalar sweep and batch checks pass:

```sh
cargo add rayon@1.12
```

You create one pool for the build and use it across the layers. You divide the
result vector with `par_chunks_mut` and align each output chunk with its input
pairs. Each chunk calls the batch kernel and returns counts or an error. You
validate finite results and the same `1 + 1e-9` value limit before
commit. You commit after the pool's layer work succeeds. You keep a direct
path for one worker or a small layer to avoid task overhead.

You can consult [Rayon's parallel-slice API](https://docs.rs/rayon/1.12.0/rayon/slice/trait.ParallelSliceMut.html)
and [pool configuration](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPoolBuilder.html).
You can use scoped standard-library threads to explore borrowing, but creating
operating-system threads for each of 1201 layers adds work to the timed solve.

You can begin with the C scheduler's chunk size. For `K` classes and `T` workers,
it chooses `2048 * max(1, ceil(K / (2048 * 4 * T)))`; one-worker execution uses
the whole layer. You treat this as a benchmark starting point. You need no lock
for each table read or each solved value in the read/commit design.

You compare full results with one worker and several workers. You require the
same class count and successful independent certificates. You can expect bitwise
agreement if per-class arithmetic and input tables stay unchanged; you should
investigate any dependence on worker count.

## 8. Verification and output

You verify before you replace an output file. You require a complete class count
and finite values in `[-1 - 1e-9, 1 + 1e-9]`, plus a terminal column of `-1`.
You check the following six reference values with absolute tolerance `1e-6`:

| `(checker ST, checker TTD, dropper ST, dropper TTD)` | Value |
| --- | --- |
| `(0, 0, 0, 0)` | `0.08985007280951046` |
| `(240, 0, 240, 0)` | `0.3372132166291093` |
| `(10, 60, 200, 0)` | `-0.7944428916469297` |
| `(150, 90, 30, 120)` | `0.7244093036356785` |
| `(250, 300, 40, 0)` | `0.9981152817381969` |
| `(100, 140, 100, 140)` | `0.1877386378276193` |

You reconstruct 1200 state classes at uniform intervals and check their full
matrix certificates through a separate scalar calculation. Divide after each
unscaled dot product in this recheck to reduce shared implementation mistakes.
For an initial release candidate, compare the complete table against the current
C output and audit full certificates outside the benchmark.

You distinguish a local stage gap from full-game accuracy. With stage solve
error `e`, assembly error `a`, and largest child error `E`, you budget `e + a + E`.
The paper's nominal uniform bound at the root is `0.0006005` in value. The
measured full-product gate uses floating endpoints; a formal enclosure at its
threshold needs endpoint rounding bounds. The sum path has its explicit radius.

You test failure behavior before output: inject an error after one successful
evaluation in a layer and check that you have committed no values from that
layer. Test an invalid second result as well as a callback error. A successful
empty layer must leave the table unchanged.

You can implement a small NumPy writer through the documented
[NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
For this table, use row-major shape `(17011, 17012)` and little-endian float64
data. You include the format magic and version, header length, and a padded
newline-terminated header. Follow the selected NPY version's header encoding
and length field. Record `descr: '<f8'`, `fortran_order: False`, and the exact
shape in the header. Do not infer byte order from your current processor.
You can use `f64::to_le_bytes` with a buffered writer to preserve a safe
implementation; measure serialization as part of output time.
You can choose a format crate if writing this header distracts from the solver.

You first test a small array with distinct values and reopen it with `np.load`.
For the real file, write to a temporary path in the destination directory, flush
and close it, then rename it over the requested path after successful verification.
You preserve the previous file if verification or writing fails. You treat
power-loss durability as a separate contract if you decide to add syncing.

You should save the Rust table as `../build/V-rust.npy` during development so you
retain `../build/V.npy` for comparison. From the repository root, you can use
this verification command after your writer works:

```sh
uv run python - <<'PY'
import sys
import numpy as np
sys.path.insert(0, 'src')
import dth

actual = np.load('build/V-rust.npy', mmap_mode='r')
reference = np.load('build/V.npy', mmap_mode='r')
assert actual.shape == reference.shape == (17011, 17012)
assert actual.dtype == reference.dtype == np.dtype('<f8')
dth.verify(actual, dth.build_table())
largest_difference = 0.0
for start in range(0, 17011, 128):
    block = actual[start:start + 128]
    assert np.isfinite(block).all()
    assert np.max(np.abs(block)) <= 1 + 1e-9
    assert np.all(block[:, -1] == -1)
    reference_block = reference[start:start + 128]
    assert np.isfinite(reference_block).all()
    delta = np.max(np.abs(block - reference_block))
    largest_difference = max(largest_difference, float(delta))
assert largest_difference <= 1e-6
print(f'You passed verification; maximum reference difference: {largest_difference:.3g}')
PY
```

You check that the reference contains finite values before computing differences;
a NaN reference can otherwise hide a failed maximum-difference check.
You keep the sampled check and whole-table comparison distinct from a complete
independent certificate audit. The command above performs the first two checks.

## 9. Performance and completion

You aim to reproduce the current solver's algorithm before you tune processor
details. On September 5, 2026, the C/Python build took a median 8.01 seconds for
the complete command on your M5 Pro with 15 threads. That measurement includes
verification and saving. You have no Rust timing until you finish this crate.
You can find the recorded baseline in [the repository README](../README.md).

You can build for your processor after correctness checks pass:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release
```

You use Cargo's release optimization with thin LTO and one codegen unit in the
starter manifest. You evaluate these settings through complete-build timings
and compare them with Cargo defaults. See [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html).

You can implement a command interface with `--threads`, `--stop-at`, and
`--output` during the sweep and output milestones. For a completed build, your
measurement command can be:

```sh
/usr/bin/time -p target/release/solver --threads 15 --stop-at 0 --output ../build/V-rust.npy
```

You record the complete wall time and the time spent in each major phase.
Include table construction and allocation, along with verification and writing.
Report compilation time as a separate measurement if you compare a cold build.
You should retain several complete runs and compare their median and range.

You can compare the routes against the recorded C build:

| Route | Recorded class count |
| --- | ---: |
| Pure | 334,177 |
| Sum certificate | 288,987,611 |
| Full mixed certificate | 52,333 |
| Total | 289,374,121 |

You require the total count. You use route differences to investigate numerical
changes, since a value near a gate can change route with arithmetic order.
You do not require a bitwise match across C and Rust math libraries.

You can test batch widths 8 and 16, worker counts below and at the available
core count, and chunk sizes around the initial scheduler choice. Keep each
experiment separate so you can attribute a timing change. You should inspect
vectorization or bounds checks when timing identifies the kernel as a limit.
You keep scalar and independent certificate checks after arithmetic changes.

You should delete per-class allocation and redundant products before adding
intrinsics. You keep the full-product fallback and finite checks because they
carry the acceptance contract. You add complexity when a measured result
justifies it and you can state its correctness argument.

You have finished when your full Rust build passes the table checks and preserves
the previous output on failure. You can claim a sub-10-second Rust solve after
you measure the complete verified command below that threshold on your machine.

## Your implementation notes

You can record decisions as you complete the milestones:

| Decision | Your reason and evidence |
| --- | --- |
| Profile representation | |
| Stage error and certificate representation | |
| Arithmetic environment contract | |
| Layer result staging | |
| Batch width and worker count | |
| Complete-build timing | |
