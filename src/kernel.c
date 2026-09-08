/* We solve 16 classes per batch with thread-local scratch.
   We read V as rows of ldv doubles and succ as rows of 60 int32 entries.
   We reserve column ldv-1 for terminal wins, with value -1. */
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <fenv.h>

#if defined(__FAST_MATH__) || FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "We require IEEE binary64 without fast-math."
#endif

#define N 60
#define L 16
#define GAP 1e-6
#define CERT_RADIUS 1e-10
_Static_assert(N == 60 && 2 * CERT_RADIUS <= GAP, "Recheck the bound in certificate.md.");

/* We write certified values and count pure and equalizer results.
   We return the failure count, -1 for an out-of-range result, -2 for invalid input,
   or -3 for an unsupported floating-point environment. */
long solve_chunk(long n, const int32_t *pcs, const int32_t *pds, double *V, long ldv,
                 const int32_t *succ, const int32_t *fail, const double *rev, int64_t *counts)
{
    volatile double normal_min = DBL_MIN;
    if (fegetround() != FE_TONEAREST || normal_min * 0.5 == 0.0) return -3;
    double S[N][L], R[N][L], Q[N][L];
    double F[L], D0[L], MN[L], MX[L], SUM[L];
    int nonnegative[L], bounded[L];
    long failed = 0;
    for (long base = 0; base < n; base += L) {
        long m = n - base < L ? n - base : L;
        /* We fill spare lanes with copies of the first class. */
        for (int l = 0; l < L; l++) {
            long i = base + (l < m ? l : 0);
            const double *row = V + (long)pds[i] * ldv;
            const int32_t *ch = succ + (long)pcs[i] * N;
            double lo = INFINITY, hi = -INFINITY;
            for (int k = 0; k < N; k++) {
                double s = -row[ch[k]];
                if (!isfinite(s)) return -2;
                S[k][l] = s;
                if (s < lo) lo = s;
                if (s > hi) hi = s;
            }
            double p = rev[pcs[i]];
            if (!(p >= 0.0 && p <= 1.0)) return -2;
            F[l] = p * (-row[fail[pcs[i]]]) + (1.0 - p);
            if (!isfinite(F[l])) return -2;
            double s0 = S[0][l], f = F[l];
            MN[l] = fmax(lo, fmin(f, s0));
            MX[l] = fmin(hi, fmax(f, s0));
            D0[l] = s0 - f;
            bounded[l] = lo >= -2.0 && hi <= 2.0 && fabs(f) <= 2.0;
        }
        /* We reuse Q for coefficients b[k] = (S[k-1] - S[k]) / d0.
           We solve r[k] = sum(b[k-j] * r[j]) across independent lanes. */
        for (int k = 1; k < N; k++)
            for (int l = 0; l < L; l++)
                Q[k][l] = (S[k - 1][l] - S[k][l]) / (D0[l] != 0.0 ? D0[l] : 1.0);
        for (int l = 0; l < L; l++) R[0][l] = 1.0;
        for (int k = 1; k < N; k++) {
            double acc[L] = {0};
            for (int j = 0; j < k; j++)
                for (int l = 0; l < L; l++) acc[l] += Q[k - j][l] * R[j][l];
            for (int l = 0; l < L; l++) R[k][l] = acc[l];
        }
        /* We reverse and normalize the nonnegative weights for the Checker. */
        for (int l = 0; l < L; l++) { SUM[l] = 0.0; nonnegative[l] = 1; }
        for (int k = 0; k < N; k++)
            for (int l = 0; l < L; l++) {
                double r = R[N - 1 - k][l];
                double q = r > 0.0 ? r : 0.0;
                nonnegative[l] &= r >= 0.0;
                if (!isfinite(r)) q = NAN;
                Q[k][l] = q;
                SUM[l] += q;
            }
        /* We check both bounds through the reversal symmetry of M. */
        for (int l = 0; l < m; l++) {
            double v;
            int kind;
            if (MX[l] - MN[l] <= GAP) {
                v = 0.5 * (MN[l] + MX[l]);
                kind = 0;
            } else {
                int ok = fabs(D0[l]) >= 1e-12 && isfinite(SUM[l]) && SUM[l] > 0.0;
                if (!ok) { failed++; continue; }
                /* We enclose both saddle bounds in v +/- CERT_RADIUS; see certificate.md. */
                if (nonnegative[l] && bounded[l] && fabs(D0[l]) >= GAP) {
                    v = F[l] + D0[l] / SUM[l];
                } else {
                    /* We check the full matrix after clipping negative weights. */
                    for (int k = 0; k < N; k++) Q[k][l] /= SUM[l];
                    double cum = 0.0, mn = INFINITY, mx = -INFINITY, f = F[l];
                    for (int k = 0; k < N; k++) {
                        double x = 0.0;
                        for (int j = 0; j < N - k; j++) x += S[j][l] * Q[k + j][l];
                        x += f * cum;
                        if (!isfinite(x)) ok = 0;
                        if (x < mn) mn = x;
                        if (x > mx) mx = x;
                        cum += Q[k][l];
                    }
                    if (!(ok && fabs(cum - 1.0) <= 1e-12 && mx - mn <= GAP)) {
                        failed++;
                        continue;
                    }
                    v = 0.5 * (mn + mx);
                }
                kind = 1;
            }
            if (!(fabs(v) <= 1.0 + 1e-9)) return -1;
            V[(long)pcs[base + l] * ldv + pds[base + l]] = v;
            counts[kind]++;
        }
    }
    return failed;
}
