# DTH

We compute one game value for each of the 289,374,121 Drop the Handkerchief state classes in a backward pass over the potential layers. You can find the rules and error bounds in `paper/dth.tex`.

You can build and verify the table with:

    uv run src/dth.py

You get `build/V.npy`, a 2.3 GB float64 array. You keep the previous file if verification or writing fails.

You can run the tests with:

    uv run pytest

We use `src/dth.py` for the profile tables and layer schedule. We use `src/kernel.c` for 16-class batches, with precomputed recurrence coefficients and local accumulators. We reuse three scratch arrays across the calculation phases. You need a C compiler; we compile the kernel on first use or after a source change with `cc -O3 -march=native`.

For finite nonnegative weights, we calculate the value from their sum and enclose both saddle bounds with a radius of `1e-10`. You can read the proof and its checked assumptions in [src/certificate.md](src/certificate.md). We retain the full matrix check for clipped weights or inputs outside those assumptions. We reject invalid child values and overflowed weight sums.

We require a stage-game saddle gap of at most `1e-6`. Before saving, we check the solved count and six reference values, and re-solve 1,200 sampled classes with a NumPy certificate. We abort on a failed check. You should use the paper's accumulated error bound to interpret full-game accuracy.

On September 5, 2026, we measured three complete `uv run src/dth.py` commands on an Apple M5 Pro with 15 threads and Apple Clang 21. We measured a median wall time of **8.01 seconds**, including verification and saving. We measured 8.64 seconds for the run that compiled the kernel. You can include those costs in your measurement with `/usr/bin/time -p uv run src/dth.py`.

We used the weight-sum certificate for 288,987,611 classes and the full matrix certificate for 52,333 mixed classes. We retained 334,177 pure results. Across the complete tables, we found a maximum difference of `2.0e-15` from the previous solver. We passed 66 tests, including 90-digit Decimal comparisons, and checked the C kernel with AddressSanitizer and UndefinedBehaviorSanitizer. We rechecked all 289,374,121 saved values with the original full-matrix certificate.

You can generate the paper's figure data from a finished table with `uv run paper/generate_figure_data.py`.
