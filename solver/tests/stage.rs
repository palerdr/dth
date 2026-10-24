use solver::stage::{ACTIONS, MATRIX_SIZE, MatrixError, StageMatrix, Strat};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "You returned {actual:?}; expected {expected:?}."
    );
}

fn point_mass(action: usize) -> Strat {
    let mut probabilities = [0.0; ACTIONS];
    probabilities[action] = 1.0;
    probabilities
}

#[test]
fn e08_constructor_rejects_nonfinite_payoffs() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in [0, ACTIONS - 1] {
            let mut s = [0.0; ACTIONS];
            s[index] = bad;
            assert!(matches!(
                StageMatrix::new(s, 0.0),
                Err(MatrixError::SuccessError)
            ));
        }
        assert!(matches!(
            StageMatrix::new([0.0; ACTIONS], bad),
            Err(MatrixError::FailError(value)) if value == bad || (value.is_nan() && bad.is_nan())
        ));
    }
}

#[test]
fn e08_dense_matrix_preserves_orientation_and_finite_synthetic_payoffs() {
    let s = std::array::from_fn(|i| i as f64 - 2.0);
    let matrix = StageMatrix::new(s, -3.0).unwrap().matrix();
    assert_eq!(&matrix[..ACTIONS], &s);
    assert_eq!(matrix[ACTIONS], -3.0);
    assert_eq!(&matrix[ACTIONS + 1..2 * ACTIONS], &s[..ACTIONS - 1]);
    assert_eq!(
        &matrix[MATRIX_SIZE - ACTIONS..MATRIX_SIZE - 1],
        &[-3.0; ACTIONS - 1]
    );
    assert_eq!(matrix[MATRIX_SIZE - 1], s[0]);
}

#[test]
fn e09_certificate_accepts_zero_probabilities_and_bounds_the_constant_game() {
    let matrix = StageMatrix::new([0.75; ACTIONS], 0.75).unwrap().matrix();
    let bounds = StageMatrix::certificate(&matrix, &point_mass(0), &point_mass(59)).unwrap();
    assert_eq!(bounds, (0.75, 0.75));
    assert_eq!(
        StageMatrix::certificate(&matrix, &[0.0; ACTIONS], &[0.0; ACTIONS]),
        Err(MatrixError::InvalidWeightSum(0.0))
    );
}

#[test]
fn e09_certificate_rejects_nonfinite_matrix_entries_before_reductions() {
    let p = point_mass(0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in [0, MATRIX_SIZE / 2, MATRIX_SIZE - 1] {
            let mut matrix = [0.75; MATRIX_SIZE];
            matrix[index] = bad;
            assert_eq!(
                StageMatrix::certificate(&matrix, &p, &p),
                Err(MatrixError::MatrixFiniteError)
            );
        }
    }
}

#[test]
fn e09_certificate_rejects_invalid_probabilities_for_both_players() {
    let matrix = [0.75; MATRIX_SIZE];
    let valid = point_mass(0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5] {
        let mut invalid = valid;
        invalid[3] = bad;
        for (p, q) in [(&invalid, &valid), (&valid, &invalid)] {
            let error = StageMatrix::certificate(&matrix, p, q).unwrap_err();
            if bad.is_finite() {
                assert_eq!(error, MatrixError::NegativeWeight(3));
            } else {
                assert_eq!(error, MatrixError::NonFiniteWeight(3));
            }
        }
    }
}

#[test]
fn e09_certificate_enforces_probability_sum_tolerance_for_both_players() {
    let matrix = [0.75; MATRIX_SIZE];
    let valid = point_mass(0);
    for mass in [0.5, 2.0, 1.0 - 2e-12, 1.0 + 2e-12] {
        let mut invalid = valid;
        invalid[0] = mass;
        for (p, q) in [(&invalid, &valid), (&valid, &invalid)] {
            assert_eq!(
                StageMatrix::certificate(&matrix, p, q),
                Err(MatrixError::InvalidProbabilitySum(mass))
            );
        }
    }
    for mass in [1.0 - 5e-13, 1.0 + 5e-13] {
        let mut near_unit = valid;
        near_unit[0] = mass;
        for (p, q) in [(&near_unit, &valid), (&valid, &near_unit)] {
            assert!(StageMatrix::certificate(&matrix, p, q).is_ok());
        }
    }
    let overflow = [f64::MAX; ACTIONS];
    for (p, q) in [(&overflow, &valid), (&valid, &overflow)] {
        assert_eq!(
            StageMatrix::certificate(&matrix, p, q),
            Err(MatrixError::InvalidWeightSum(f64::INFINITY))
        );
    }
}

#[test]
fn e09_certificate_rejects_nonfinite_products_of_finite_inputs() {
    let matrix = [f64::MAX; MATRIX_SIZE];
    let valid = point_mass(0);
    let mut near_unit = valid;
    near_unit[0] += 5e-13;
    for (p, q) in [(&near_unit, &valid), (&valid, &near_unit)] {
        assert_eq!(
            StageMatrix::certificate(&matrix, p, q),
            Err(MatrixError::CertificateFiniteError)
        );
    }
}

#[test]
fn e09_e10_e11_e16_structured_formulas_match_dense_reference_products() {
    for case in 0..32 {
        let s = std::array::from_fn(|i| 0.7 * (i as f64 * 0.19 + case as f64 * 0.31).sin());
        let f = 0.6 * (case as f64 * 0.17).cos();
        let stage = StageMatrix::new(s, f).unwrap();
        let matrix = stage.matrix();
        let p: Strat = std::array::from_fn(|i| (i + 1) as f64 / 1830.0);
        let mut q: Strat = std::array::from_fn(|i| ((i * 17 + 13) % 71 + 1) as f64);
        let sum: f64 = q.iter().sum();
        for probability in &mut q {
            *probability /= sum;
        }

        let row_payoffs: Vec<f64> = matrix
            .chunks_exact(ACTIONS)
            .map(|row| {
                row.iter()
                    .zip(&q)
                    .map(|(value, probability)| value * probability)
                    .sum()
            })
            .collect();
        let column_payoffs: Vec<f64> = (0..ACTIONS)
            .map(|c| {
                matrix
                    .iter()
                    .skip(c)
                    .step_by(ACTIONS)
                    .zip(&p)
                    .map(|(value, probability)| value * probability)
                    .sum()
            })
            .collect();
        for (actual, expected) in stage.structured_payoffs(&q).iter().zip(&row_payoffs) {
            assert_close(*actual, *expected);
        }
        for (actual, rows) in stage.drift_vector(&q).iter().zip(row_payoffs.windows(2)) {
            assert_close(*actual, rows[0] - rows[1]);
        }
        let (lo, hi) = StageMatrix::certificate(&matrix, &p, &q).unwrap();
        assert_close(
            lo,
            column_payoffs.iter().copied().fold(f64::INFINITY, f64::min),
        );
        assert_close(
            hi,
            row_payoffs
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        );

        let max_row_min = matrix
            .chunks_exact(ACTIONS)
            .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
            .fold(f64::NEG_INFINITY, f64::max);
        let min_column_max = (0..ACTIONS)
            .map(|c| {
                matrix
                    .iter()
                    .skip(c)
                    .step_by(ACTIONS)
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .fold(f64::INFINITY, f64::min);
        let (lo, hi) = stage.pure_bounds();
        assert_close(lo, max_row_min);
        assert_close(hi, min_column_max);
    }
}

#[test]
fn e12_e17_recurrences_reject_zero_difference_without_rejecting_the_stage() {
    for value in [0.0, 0.25] {
        let stage = StageMatrix::new([value; ACTIONS], value).unwrap();
        assert_eq!(stage.pure_bounds(), (value, value));
        assert_eq!(stage.weights(), Err(MatrixError::ZeroDifference));
        assert_eq!(stage.scaled_weights(), Err(MatrixError::ZeroDifference));
    }
}

#[test]
fn e12_e17_recurrences_match_geometric_weights() {
    let mut s = [0.25; ACTIONS];
    s[0] = -0.25;
    let stage = StageMatrix::new(s, 0.0).unwrap();
    let expected: Strat = std::array::from_fn(|k| 2.0_f64.powi(k as i32));
    assert_eq!(stage.weights().unwrap(), expected);
    assert_eq!(stage.scaled_weights().unwrap(), expected);
}

#[test]
fn e12_e17_signed_coefficients_match_the_closed_form() {
    for a in [0.1_f64, 0.9, 1.0, 1.2] {
        let mut s = [0.05 + 0.4 * a - 0.2 * a * a; ACTIONS];
        s[0] = 0.05;
        s[1] = 0.05 + 0.4 * a;
        let stage = StageMatrix::new(s, 0.25).unwrap();
        let reference = stage.weights().unwrap();
        let scaled = stage.scaled_weights().unwrap();
        for k in 0..ACTIONS {
            let expected = (k + 1) as f64 * a.powi(k as i32);
            let tolerance = 1e-14 + 1e-11 * expected.abs();
            assert!((reference[k] - expected).abs() <= tolerance);
            assert!((scaled[k] - expected).abs() <= tolerance);
            assert!((reference[k] - scaled[k]).abs() <= tolerance);
        }
    }
}

#[test]
fn e13_equalizer_pair_normalizes_nonnegative_weights_and_reverses_the_mix() {
    let mut weights = [0.0; ACTIONS];
    weights[0] = 1.0;
    weights[1] = 2.0;
    weights[59] = 3.0;
    let (p, q) = StageMatrix::equalizer_pair(&weights).unwrap();
    assert_close(p[0], 1.0 / 6.0);
    assert_close(p[1], 1.0 / 3.0);
    assert_eq!(p[59], 0.5);
    assert!(p[2..59].iter().all(|&value| value == 0.0));
    assert_close(p.iter().sum(), 1.0);
    assert_close(q.iter().sum(), 1.0);
    for i in 0..ACTIONS {
        assert_eq!(p[i], q[ACTIONS - 1 - i]);
    }
}

#[test]
fn e13_equalizer_pair_rejects_nonfinite_and_negative_weights() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.5] {
        let mut weights = point_mass(0);
        weights[1] = bad;
        let error = StageMatrix::equalizer_pair(&weights).unwrap_err();
        if bad.is_finite() {
            assert_eq!(error, MatrixError::NegativeWeight(1));
        } else {
            assert_eq!(error, MatrixError::NonFiniteWeight(1));
        }
    }
}

#[test]
fn e13_equalizer_pair_rejects_zero_and_overflowed_sums() {
    assert_eq!(
        StageMatrix::equalizer_pair(&[0.0; ACTIONS]),
        Err(MatrixError::InvalidWeightSum(0.0))
    );
    assert_eq!(
        StageMatrix::equalizer_pair(&[f64::MAX; ACTIONS]),
        Err(MatrixError::InvalidWeightSum(f64::INFINITY))
    );
}

#[test]
fn e14_exact_value_matches_the_uniform_equalizer_certificate() {
    let mut s = [0.0; ACTIONS];
    s[0] = -0.25;
    let stage = StageMatrix::new(s, 0.0).unwrap();
    let weights = stage.weights().unwrap();
    assert_eq!(weights, [1.0; ACTIONS]);
    let value = stage.exact_values(&weights);
    assert_close(value, -0.25 / 60.0);
    let (p, q) = StageMatrix::equalizer_pair(&weights).unwrap();
    let (lo, hi) = StageMatrix::certificate(&stage.matrix(), &p, &q).unwrap();
    assert_close(lo, value);
    assert_close(hi, value);
}

#[test]
fn e14_recurrence_and_certificate_allow_partial_support() {
    let mut s = [1.0; ACTIONS];
    s[0] = 0.0;
    s[1] = 0.0;
    let stage = StageMatrix::new(s, 1.0).unwrap();
    let expected: Strat = std::array::from_fn(|k| if k % 2 == 0 { 1.0 } else { 0.0 });
    assert_eq!(stage.weights().unwrap(), expected);
    assert_eq!(stage.scaled_weights().unwrap(), expected);
    let (p, q) = StageMatrix::equalizer_pair(&expected).unwrap();
    let value = stage.exact_values(&expected);
    let (lo, hi) = StageMatrix::certificate(&stage.matrix(), &p, &q).unwrap();
    assert_close(value, 29.0 / 30.0);
    assert_close(lo, value);
    assert_close(hi, value);
}

#[test]
fn e15_clipped_pair_rejects_nonfinite_raw_weights_before_clipping() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut weights = point_mass(0);
        weights[1] = bad;
        assert_eq!(
            StageMatrix::clipped_pair(&weights),
            Err(MatrixError::NonFiniteWeight(1))
        );
    }
}

#[test]
fn e15_clipped_pair_rejects_zero_and_overflowed_clipped_sums() {
    for weights in [[0.0; ACTIONS], [-1.0; ACTIONS]] {
        assert_eq!(
            StageMatrix::clipped_pair(&weights),
            Err(MatrixError::InvalidWeightSum(0.0))
        );
    }
    assert_eq!(
        StageMatrix::clipped_pair(&[f64::MAX; ACTIONS]),
        Err(MatrixError::InvalidWeightSum(f64::INFINITY))
    );
}

#[test]
fn e15_clipped_candidates_need_the_full_certificate() {
    for (last_success, accepted) in [(-1e-7, true), (-0.1, false)] {
        let mut s = [0.5; ACTIONS];
        s[0] = 0.0;
        s[ACTIONS - 1] = last_success;
        let stage = StageMatrix::new(s, 0.5).unwrap();
        let raw = stage.scaled_weights().unwrap();
        assert!(raw[ACTIONS - 1] < 0.0);
        assert!(stage.weights().unwrap()[ACTIONS - 1] < 0.0);
        let (p, q) = StageMatrix::clipped_pair(&raw).unwrap();
        assert_eq!(p[ACTIONS - 1], 0.0);
        assert_close(p.iter().sum(), 1.0);
        assert_close(q.iter().sum(), 1.0);
        let (lo, hi) = StageMatrix::certificate(&stage.matrix(), &p, &q).unwrap();
        assert_eq!(hi - lo <= 1e-6, accepted);
        let row_payoffs = stage.structured_payoffs(&q);
        assert_close(
            lo,
            row_payoffs.iter().copied().fold(f64::INFINITY, f64::min),
        );
        assert_close(
            hi,
            row_payoffs
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        );
    }
}
