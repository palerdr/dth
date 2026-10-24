use thiserror::Error;

use crate::profiles::{
    PROFILE_COUNT, Profile, profile_catalog, profile_id, revival_probability, revived_child,
    success_child,
};
use crate::stage::{ACTIONS, MatrixError, StageMatrix};
use crate::sweep::{build_buckets, layer_pairs};

/// You store the value from the current Dropper's perspective at `V[pc, pd]`.
/// Each row holds `n + 1` cells; column `n` is the terminal `WIN` sentinel
/// with value `-1`, so a gather negates it into `+1` for the Dropper.
pub struct Table {
    pub n: usize,
    pub values: Vec<f64>,
}

/// You precompute the ID-indexed transitions once from the catalog.
pub struct Transitions {
    pub profiles: Vec<Profile>,
    pub succ: Vec<[u32; ACTIONS]>,
    pub fail: Vec<u32>,
    pub rev: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum TableError {
    #[error("You cannot size a table with {0} profiles.")]
    Overflow(usize),
    #[error("Layer {layer}, class ({pc}, {pd}): {source}")]
    Stage {
        layer: usize,
        pc: usize,
        pd: usize,
        source: MatrixError,
    },
}

impl Transitions {
    pub fn build() -> Self {
        let profiles = profile_catalog().expect("The catalog stays in the valid domain.");
        let win = profiles.len() as u32;
        let mut succ = Vec::with_capacity(profiles.len());
        let mut fail = Vec::with_capacity(profiles.len());
        let mut rev = Vec::with_capacity(profiles.len());
        for profile in &profiles {
            let mut row = [win; ACTIONS];
            for (k, cell) in row.iter_mut().enumerate() {
                let child = success_child(profile, k + 1).expect("Lag stays in 1..=60.");
                if let Some(child) = child {
                    *cell = profile_id(&child);
                }
            }
            succ.push(row);
            let revived = revived_child(profile).expect("The revived child stays in the domain.");
            fail.push(revived.map_or(win, |child| profile_id(&child)));
            rev.push(match *profile {
                Profile::Eligible { st, ttd } => revival_probability(st, ttd),
                Profile::FailureFatal { .. } => 0.0,
            });
        }
        debug_assert_eq!(profiles.len(), PROFILE_COUNT);
        Self {
            profiles,
            succ,
            fail,
            rev,
        }
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

impl Table {
    /// You allocate `n * (n + 1)` cells: NaN for every class and `-1` in column `n`.
    pub fn new(n: usize) -> Result<Self, TableError> {
        let cells = n
            .checked_add(1)
            .and_then(|stride| n.checked_mul(stride))
            .ok_or(TableError::Overflow(n))?;
        let mut values = vec![f64::NAN; cells];
        for row in values.chunks_exact_mut(n + 1) {
            row[n] = -1.0;
        }
        Ok(Self { n, values })
    }

    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.n + 1
    }

    #[inline(always)]
    pub fn index(&self, pc: usize, pd: usize) -> usize {
        pc * self.stride() + pd
    }

    pub fn get(&self, pc: usize, pd: usize) -> f64 {
        self.values[self.index(pc, pd)]
    }

    /// You read the Dropper's row `pd` through the Checker's transitions:
    ///
    /// ```text
    /// S[k] = -V[pd, succ(pc, k)]
    /// F    = rev(pc) * (-V[pd, fail(pc)]) + (1 - rev(pc))
    /// ```
    pub fn gather(
        &self,
        t: &Transitions,
        pc: usize,
        pd: usize,
    ) -> Result<StageMatrix, MatrixError> {
        let start = pd * self.stride();
        let row = &self.values[start..start + self.stride()];
        let mut s = [0.0; ACTIONS];
        for (cell, &child) in s.iter_mut().zip(&t.succ[pc]) {
            *cell = -row[child as usize];
        }
        let p = t.rev[pc];
        let f = p * (-row[t.fail[pc] as usize]) + (1.0 - p);
        StageMatrix::new(s, f)
    }

    /// You solve layers from 1200 down to `stop_at`, committing each layer after
    /// every class in it has a value. Layers below `stop_at` stay NaN.
    pub fn sweep(&mut self, t: &Transitions, stop_at: usize) -> Result<usize, TableError> {
        let buckets = build_buckets(&t.profiles);
        let mut results: Vec<f64> = Vec::new();
        let mut solved = 0;
        for layer in (stop_at..=1200).rev() {
            let pairs = layer_pairs(&buckets, layer);
            results.clear();
            for &(pc, pd) in &pairs {
                let value = self
                    .gather(t, pc, pd)
                    .and_then(|stage| stage.value())
                    .map_err(|source| TableError::Stage {
                        layer,
                        pc,
                        pd,
                        source,
                    })?;
                results.push(value);
            }
            for (&(pc, pd), &value) in pairs.iter().zip(&results) {
                let idx = self.index(pc, pd);
                self.values[idx] = value;
            }
            solved += pairs.len();
        }
        Ok(solved)
    }
}
