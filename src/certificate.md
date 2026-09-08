# Certificate from nonnegative weights

We use this bound in `kernel.c` to avoid a second triangular matrix product.
We retain the full matrix check after clipping a negative weight.

We treat the gathered binary64 payoffs as the exact entries of the stage
matrix. You must account for child-value error through the accumulated bound
in `paper/dth.tex`.

We require 60 actions, round-to-nearest binary64 arithmetic with gradual
underflow, and compilation without fast-math. We check the rounding mode and
underflow behavior at entry to the kernel. We use the shortcut for finite,
nonnegative computed weights with a finite positive sum, payoffs in
`[-2, 2]`, and `abs(D0) >= 1e-6`.

We store the center `v` of an interval with radius `CERT_RADIUS = 1e-10`.
We prove that this interval encloses both saddle bounds. Its width is
`2e-10`, within the required `1e-6` gate.

## Recurrence residual

We index actions from zero. Let

\[
d=S_0-F,\quad \Delta_m=S_m-S_{m-1},\quad
\widehat d=\operatorname{fl}(d),\quad
\widehat b_m=\operatorname{fl}
 \left(\operatorname{fl}(-\Delta_m)/\widehat d\right).
\]

We compute `r[0] = 1` and

\[
r_k=\operatorname{fl}\left(\sum_{j=0}^{k-1}
                  \widehat b_{k-j}r_j\right).
\]

We use the standard floating-point model with unit roundoff
\(u=2^{-53}\) and \(\gamma_n=nu/(1-nu)\). We bound dot-product error
by \(\gamma_{2k}\sum|a_jb_j|\); this bound permits separate multiply
and add operations or fused multiply-add operations. You can find the
summation analysis in [Higham, The Accuracy of Floating Point Summation](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)
and dot-product analysis in [Castaldo, Whaley, and Chronopoulos, Reducing Floating Point Error in Dot Product Using the Superblock Family of Algorithms](https://www.cs.utsa.edu/~atc/pub/J42.pdf).

We apply the dot-product bound to the computed weights as fixed inputs.
We do not need a forward-error bound on the whole triangular solve.
Define the exact sum of those weights as \(W=\sum_{j=0}^{59}r_j\).
We require \(r_j\ge0\), which gives \(W\ge1\) because \(r_0=1\).

For each row difference, we expand its residual as

\[
\begin{aligned}
d r_k+\sum_{j<k}\Delta_{k-j}r_j
={}&\widehat d\left(r_k-\sum_{j<k}\widehat b_{k-j}r_j\right)\\
 &+(d-\widehat d)r_k
   +\sum_{j<k}(\widehat d\widehat b_{k-j}+\Delta_{k-j})r_j.
\end{aligned}
\]

We divide by \(W\). Since \(|d|,|\Delta_m|\le4\), and any subset
of the nonnegative weights sums to at most \(W\), we obtain

\[
\frac{\left|d r_k+\sum_{j<k}\Delta_{k-j}r_j\right|}{W}
\le4\gamma_{118}(1+\gamma_2)+4u+4\gamma_2+500\eta
<6\times10^{-14},
\]

where \(\eta=2^{-1075}\) bounds the absolute rounding error of an
underflowing operation. We include the absolute underflow terms from
coefficient formation and the dot product. We use \(W\ge1\) to bound
those terms after division. We exclude intermediate overflow under
round-to-nearest by checking the recurrence results: we cannot obtain a
finite result from an infinity through these multiply-add operations.

## Saddle bounds and stored value

We choose the exact strategies
\(p_j=r_j/W\) and \(q_j=r_{59-j}/W\).
We identify the residual above with an adjacent-row difference of \(Mq\).
Across 59 differences, we get

\[
\max_i(Mq)_i-\min_i(Mq)_i < 59(6\times10^{-14})=3.54\times10^{-12}.
\]

We use the reversal symmetry of the Toeplitz stage matrix to obtain
\((p^TM)_j=(Mq)_{59-j}\). Thus we can use the minimum and maximum of
\(Mq\) as the lower and upper saddle bounds.

We calculate the last row's exact payoff as

\[
(Mq)_{59}=F+(S_0-F)/W,
\]

because \(q_{59}=r_0/W=1/W\). In the kernel, we evaluate

\[
\widehat W=\operatorname{fl}\left(\sum_{j=59}^{0}r_j\right),\qquad
v=\operatorname{fl}\left(F+\operatorname{fl}(\widehat d/\widehat W)\right).
\]

We bound the positive sum's relative error by \(\gamma_{60}\), plus
its absolute underflow terms. With \(|F|\le2\), \(|d|\le4\), and
\(W\ge1\), the subtraction, division, and final addition give
\(|v-(Mq)_{59}|<4\times10^{-14}\).
We enclose both saddle bounds within \(4\times10^{-12}\)
of `v`. We use the larger radius `1e-10` in the code.

We retain the existing pure-saddle check. We use the full matrix certificate
for clipped weights or inputs outside the shortcut's bounds. We reject
nonfinite weights and overflowed sums before either equalizer path can pass.

We test the shortcut against 90-digit Decimal solutions, including signed
coefficients and a subnormal result. We test clipped mixtures against the
full matrix so that clipping cannot trigger the shortcut.
