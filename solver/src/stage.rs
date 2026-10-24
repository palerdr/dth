use thiserror::Error;
use num_rational::{BigRational, Ratio};

pub const ACTIONS: usize = 60;
pub const MATRIX_SIZE: usize = ACTIONS * ACTIONS;
pub type Matrix = [f64; MATRIX_SIZE];
pub type Strat = [f64; ACTIONS];
const PROBABILITY_TOLERANCE: f64 = 1e-12;
/// You accept a pure stage when its bounds close within this gap.
pub const GAP: f64 = 1e-6;

#[derive(Debug, Error, PartialEq)]
pub enum MatrixError {
    #[error("You supplied a nonfinite success payoff.")]
    SuccessError,
    #[error("You supplied a nonfinite failure payoff: {0}.")]
    FailError(f64),
    #[error("You supplied a nonfinite matrix entry.")]
    MatrixFiniteError,
    #[error("You supplied a nonfinite weight at index {0}.")]
    NonFiniteWeight(usize),
    #[error("You supplied a negative weight at index {0}.")]
    NegativeWeight(usize),
    #[error("You supplied weights with sum {0}; use a finite positive sum.")]
    InvalidWeightSum(f64),
    #[error("You supplied probabilities with sum {0}; use a sum within 1e-12 of one.")]
    InvalidProbabilitySum(f64),
    #[error("You need S[0] != F to calculate recurrence weights.")]
    ZeroDifference,
    #[error("You computed a nonfinite certificate payoff.")]
    CertificateFiniteError,
}
pub struct StageMatrix {
    s: [f64; ACTIONS],
    f: f64,
}

impl StageMatrix {
    #[inline(always)]
    fn index(d: usize, c: usize) -> usize {
        (d * ACTIONS) + c
    }

    #[inline(always)]
    fn d_0(&self) -> f64 {
        self.s[0] - self.f
    }
    #[inline(always)]
    fn delta(&self, m: usize) -> f64 {
        self.s[m] - self.s[m - 1]
    }

    pub fn new(s: [f64; ACTIONS], f: f64) -> Result<Self, MatrixError> {
        if !s.iter().all(|x| x.is_finite()) {
            return Err(MatrixError::SuccessError);
        }
        if !f.is_finite() {
            return Err(MatrixError::FailError(f));
        }
        Ok(Self { s, f })
    }

    /// You expand the validated payoffs for dense reference checks.
    pub fn matrix(&self) -> Matrix {
        let mut data = [0.0; MATRIX_SIZE];

        for d in 0..ACTIONS {
            for c in 0..ACTIONS {
                let idx = Self::index(d, c);
                if c >= d {
                    data[idx] = self.s[c - d];
                } else {
                    data[idx] = self.f;
                }
            }
        }
        data
    }

    fn nonnegative_sum(r: &Strat) -> Result<f64, MatrixError> {
        let mut sum = 0.0;
        for (i, &weight) in r.iter().enumerate() {
            if !weight.is_finite() {
                return Err(MatrixError::NonFiniteWeight(i));
            }
            if weight < 0.0 {
                return Err(MatrixError::NegativeWeight(i));
            }
            sum += weight;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(MatrixError::InvalidWeightSum(sum));
        }
        Ok(sum)
    }

    /// You check finite inputs and probability sums before calculating (L, U).
    pub fn certificate(m: &Matrix, p: &Strat, q: &Strat) -> Result<(f64, f64), MatrixError> {
        if !m.iter().all(|value| value.is_finite()) {
            return Err(MatrixError::MatrixFiniteError);
        }
        for strategy in [p, q] {
            let sum = Self::nonnegative_sum(strategy)?;
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(MatrixError::InvalidProbabilitySum(sum));
            }
        }

        // L calculation (Min of p.T @ M)
        let mut l = f64::INFINITY;
        for c in 0..ACTIONS {
            let mut checker_val = 0.0;
            for d in 0..ACTIONS {
                checker_val += p[d] * m[Self::index(d, c)];
            }
            if !checker_val.is_finite() {
                return Err(MatrixError::CertificateFiniteError);
            }
            l = l.min(checker_val);
        }

        // U calculation (Max of M @ q)
        let mut u = f64::NEG_INFINITY;
        for d in 0..ACTIONS {
            let mut dropper_val = 0.0;
            for c in 0..ACTIONS {
                dropper_val += q[c] * m[Self::index(d, c)];
            }
            if !dropper_val.is_finite() {
                return Err(MatrixError::CertificateFiniteError);
            }
            u = u.max(dropper_val);
        }
        Ok((l, u))
    }

    pub fn pure_bounds(&self) -> (f64, f64) {
        let mut s_min = f64::INFINITY;
        for v in self.s.iter() {
            s_min = s_min.min(*v);
        }
        let l0 = s_min.max(self.f.min(self.s[0]));
        let mut s_max = f64::NEG_INFINITY;
        for v in self.s.iter() {
            s_max = s_max.max(*v);
        }
        let u0 = s_max.min(self.f.max(self.s[0]));
        (l0, u0)
    }

    /// You supply a legal Checker strategy and receive adjacent row-payoff differences.
    pub fn drift_vector(&self, q: &Strat) -> [f64; ACTIONS - 1] {
        let mut drift = [0.0; ACTIONS - 1];
        for i in 0..(ACTIONS - 1) {
            drift[i] = self.d_0() * q[i];
            for m in 1..(ACTIONS - i) {
                drift[i] += self.delta(m) * q[i + m];
            }
        }
        drift
    }

    /// You retain signed weights and divide after each reference dot product.
    /// You check nonfinite outputs when you select a stage result.
    pub fn weights(&self) -> Result<Strat, MatrixError> {
        let d0 = self.d_0();
        if d0 == 0.0 {
            return Err(MatrixError::ZeroDifference);
        }
        let mut r = [0.0; ACTIONS];
        r[0] = 1.0;
        for k in 1..ACTIONS {
            let mut acc = 0.0;
            for j in 0..k {
                acc += self.delta(k - j) * r[j];
            }
            r[k] = -acc / d0;
        }
        Ok(r)
    }

    /// You normalize finite nonnegative weights and reverse the mix for the Checker.
    pub fn equalizer_pair(r: &Strat) -> Result<(Strat, Strat), MatrixError> {
        let w = Self::nonnegative_sum(r)?;
        let mut p = [0.0; ACTIONS];
        for (p_val, &r_val) in p.iter_mut().zip(r.iter()) {
            *p_val = r_val / w;
        }
        let mut q = p;
        q.reverse();
        Ok((p, q))
    }

    /// You supply this stage's nonnegative recurrence weights with `r[0] = 1`.
    pub fn exact_values(&self, r: &Strat) -> f64 {
        self.f + self.d_0() / r.iter().sum::<f64>()
    }

    /// You clip finite weights and return a candidate pair for a full certificate.
    pub fn clipped_pair(r: &Strat) -> Result<(Strat, Strat), MatrixError> {
        let mut clipped = [0.0; ACTIONS];
        for (i, &weight) in r.iter().enumerate() {
            if !weight.is_finite() {
                return Err(MatrixError::NonFiniteWeight(i));
            }
            clipped[i] = weight.max(0.0);
        }
        Self::equalizer_pair(&clipped)
    }

    /// You supply a legal Checker strategy and evaluate the Dropper's row payoffs
    /// without a dense matrix. Row i pays `F` on every column c < i and `S[c - i]`
    /// on every column c >= i:
    ///
    /// ```text
    /// x[i] = F * sum_{c < i} q[c]  +  sum_{j = 0}^{59 - i} S[j] * q[i + j]
    /// ```
    ///
    /// The prefix term is the failure mass the Checker leaves before second i.
    /// The tail term is the success payoff by lag j = c - i.
    pub fn structured_payoffs(&self, q: &Strat) -> Strat {
        let mut x = [0.0; ACTIONS];
        let mut qf_prefix = 0.0;
        for i in 0..ACTIONS {
            x[i] = qf_prefix;
            for j in 0..(ACTIONS - i) {
                x[i] += self.s[j] * q[i + j];
            }
            qf_prefix += self.f * q[i];
        }
        x
    }

    /// You calculate the Dropper's equalizer weights from the row-difference recurrence.
    /// Subtracting row k - 1 from row k of M leaves `S[0] - F` in column k and
    /// `S[m] - S[m - 1]` in column k + m, so an equalizing q satisfies
    ///
    /// ```text
    /// d0   = S[0] - F
    /// b[m] = (S[m - 1] - S[m]) / d0          m = 1..=59
    /// r[0] = 1
    /// r[k] = sum_{j < k} b[k - j] * r[j]     k = 1..=59
    /// ```
    ///
    /// You divide once per coefficient `b[m]` and sum each dot product in ascending
    /// index order. You retain signed weights; you clip and normalize them later.
    /// You return `ZeroDifference` when d0 == 0, and you check nonfinite outputs
    /// at stage acceptance.
    pub fn scaled_weights(&self) -> Result<Strat, MatrixError> {
        let d0 = self.d_0();
        if d0 == 0.0 {
            return Err(MatrixError::ZeroDifference);
        }
        let mut b = [0.0; ACTIONS];
        let mut r = [0.0; ACTIONS];
        r[0] = 1.0;
        for m in 1..ACTIONS {
            b[m] = -self.delta(m) / d0;
        }
        for k in 1..ACTIONS {
            let mut acc = 0.0;
            for j in 0..k {
                acc += b[k - j] * r[j];
            }
            r[k] = acc;
        }
        Ok(r)
    }

    /// You calculate the stage value by plain routing, without a certificate.
    /// A pure stage returns the midpoint of its bounds. A mixed stage runs the
    /// recurrence, clips and normalizes the weights, reverses them for the
    /// Checker, and returns the midpoint of the row payoffs.
    pub fn value(&self) -> Result<f64, MatrixError> {
        let (l0, u0) = self.pure_bounds();
        if u0 - l0 <= GAP {
            return Ok(0.5 * (l0 + u0));
        }
        let r = self.scaled_weights()?;
        let (_, q) = Self::clipped_pair(&r)?;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for x in self.structured_payoffs(&q) {
            if !x.is_finite() {
                return Err(MatrixError::CertificateFiniteError);
            }
            mn = mn.min(x);
            mx = mx.max(x);
        }
        Ok(0.5 * (mn + mx))
    }

    pub fn exact_residuals(&self, r: &Strat) -> Vec<BigRational> {
        let s_exact: Vec<BigRational> = self.s
            .iter()
            .map(|&val| Ratio::from_float(val).expect("finite f64 expected"))
            .collect();
            
        let f_exact = Ratio::from_float(self.f).expect("finite f64 expected");
        
        let r_exact: Vec<BigRational> = r
            .iter()
            .map(|&val| Ratio::from_float(val).expect("finite f64 expected"))
            .collect();

        let w_sum: BigRational = r_exact.iter().cloned().sum();
        let d0 = &s_exact[0] - &f_exact;

        let delta = |m: usize| -> BigRational { &s_exact[m] - &s_exact[m - 1] };

        (1..ACTIONS)
            .map(|k| {
                let direct_term = &d0 * &r_exact[k];
                let convolution_term: BigRational = (0..k)
                    .map(|j| delta(k - j) * &r_exact[j])
                    .sum();

                (direct_term + convolution_term) / &w_sum
            })
            .collect()
    }
}
