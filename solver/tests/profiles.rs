use solver::profiles::{
    Profile, ValueError, canonical, eligible, potential, profile_catalog, revival_probability,
    revived_child, success_child, win_probability,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "You returned {actual:?}; expected {expected:?}."
    );
}

#[test]
fn e01_win_probability_preserves_payoff_endpoints_and_role_swap() {
    for (value, probability) in [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (0.4, 0.7)] {
        assert_close(win_probability(value).unwrap(), probability);
        assert_close(win_probability(-value).unwrap(), 1.0 - probability);
    }
}

#[test]
fn e01_win_probability_rejects_invalid_values() {
    for value in [-1.01, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            win_probability(value),
            Err(ValueError::InvalidValue(_))
        ));
    }
}

#[test]
fn e02_eligibility_matches_the_dose_rules_over_the_input_domain() {
    for st in 0..300 {
        for ttd in 0..=300 {
            let dose = st + 60;
            assert_eq!(
                eligible(st, ttd),
                dose < 300 && dose + ttd <= 300,
                "You checked ST {st}, TTD {ttd}."
            );
        }
    }
}

#[test]
fn e02_revival_probability_matches_known_values_and_fractional_exponents() {
    for (st, ttd, expected) in [
        (0, 0, 0.95),
        (120, 0, 0.475),
        (0, 60, 0.7125),
        (0, 120, 0.534375),
        (180, 60, 0.178125),
        (239, 0, 0.95 / 240.0),
        (0, 240, 0.3005859375),
        (240, 0, 0.0),
        (181, 60, 0.0),
        (0, 241, 0.0),
    ] {
        assert_close(revival_probability(st, ttd), expected);
    }
    assert_close(revival_probability(0, 90).powi(2), 0.7125 * 0.534375);
}

#[test]
fn e02_revival_probability_is_positive_exactly_inside_the_eligible_region() {
    for st in 0..300 {
        for ttd in 0..=300 {
            let probability = revival_probability(st, ttd);
            assert!(probability.is_finite());
            if st + 60 < 300 && st + 60 + ttd <= 300 {
                assert!(probability > 0.0 && probability <= 0.95);
            } else {
                assert_eq!(probability, 0.0);
            }
        }
    }
}

#[test]
fn e03_canonical_preserves_eligible_profiles_and_merges_failure_fatal_profiles() {
    for ttd in std::iter::once(0).chain(60..=300) {
        for st in 0..300 {
            let expected = if st + 60 < 300 && st + 60 + ttd <= 300 {
                Profile::Eligible { st, ttd }
            } else {
                Profile::FailureFatal { st }
            };
            assert_eq!(canonical(st, ttd).unwrap(), expected);
        }
    }
}

#[test]
fn e03_canonical_rejects_invalid_inputs_before_reduction() {
    for st in [300, usize::MAX] {
        assert!(matches!(
            canonical(st, 0),
            Err(ValueError::InvalidSt(value)) if value == st
        ));
    }
    for st in 0..300 {
        for ttd in (1..60).chain([301, usize::MAX]) {
            assert!(matches!(
                canonical(st, ttd),
                Err(ValueError::InvalidTtd(value)) if value == ttd
            ));
        }
    }
}

#[test]
fn e04_success_distinguishes_live_failure_fatal_children_from_terminal_wins() {
    for (st, ttd, lag, expected) in [
        (0, 0, 1, Some(Profile::Eligible { st: 1, ttd: 0 })),
        (239, 0, 1, Some(Profile::FailureFatal { st: 240 })),
        (239, 0, 60, Some(Profile::FailureFatal { st: 299 })),
        (180, 60, 1, Some(Profile::FailureFatal { st: 181 })),
        (0, 240, 1, Some(Profile::FailureFatal { st: 1 })),
        (250, 300, 1, Some(Profile::FailureFatal { st: 251 })),
        (0, 300, 60, Some(Profile::FailureFatal { st: 60 })),
        (240, 300, 59, Some(Profile::FailureFatal { st: 299 })),
        (240, 300, 60, None),
        (299, 300, 1, None),
        (299, 300, 60, None),
    ] {
        assert_eq!(
            success_child(&canonical(st, ttd).unwrap(), lag).unwrap(),
            expected,
            "You applied lag {lag} to raw ST {st}, TTD {ttd}."
        );
    }
}

#[test]
fn e04_success_rejects_invalid_lags_before_updating_a_profile() {
    for (st, ttd) in [(0, 0), (250, 300), (299, 300)] {
        for lag in [0, 61, usize::MAX] {
            let error = success_child(&canonical(st, ttd).unwrap(), lag).unwrap_err();
            assert!(matches!(
                &error,
                ValueError::InvalidLag(value) if *value == lag
            ));
            assert_eq!(
                error.to_string(),
                format!("You supplied lag {lag}; use 1 through 60.")
            );
        }
    }
}

#[test]
fn e04_revival_resets_st_adds_the_dose_and_reduces_the_child() {
    for (st, ttd, expected) in [
        (0, 0, Some(Profile::Eligible { st: 0, ttd: 60 })),
        (120, 60, Some(Profile::Eligible { st: 0, ttd: 240 })),
        (121, 60, Some(Profile::FailureFatal { st: 0 })),
        (180, 60, Some(Profile::FailureFatal { st: 0 })),
        (239, 0, Some(Profile::FailureFatal { st: 0 })),
        (0, 240, Some(Profile::FailureFatal { st: 0 })),
        (0, 300, None),
        (240, 300, None),
    ] {
        assert_eq!(
            revived_child(&canonical(st, ttd).unwrap()).unwrap(),
            expected,
            "You revived raw ST {st}, TTD {ttd}."
        );
    }
}

#[test]
fn e05_potential_matches_the_four_boundary_values() {
    for (profile, expected) in [
        (Profile::Eligible { st: 0, ttd: 0 }, 0),
        (Profile::Eligible { st: 180, ttd: 60 }, 240),
        (Profile::FailureFatal { st: 0 }, 301),
        (Profile::FailureFatal { st: 299 }, 600),
    ] {
        assert_eq!(potential(&profile), expected);
    }
}

#[test]
fn e04_e05_transitions_preserve_the_domain_and_increase_potential() {
    let mut profile_count = 0;
    let mut success_count = 0;
    let mut revival_count = 0;

    for ttd in std::iter::once(0).chain(60..=300) {
        for st in 0..300 {
            let can_revive = st + 60 < 300 && st + 60 + ttd <= 300;
            if !can_revive && ttd != 300 {
                continue;
            }
            profile_count += 1;
            let parent_potential = if can_revive { st + ttd } else { st + 301 };
            assert_eq!(potential(&canonical(st, ttd).unwrap()), parent_potential);

            for lag in 1..=60 {
                success_count += 1;
                let child = success_child(&canonical(st, ttd).unwrap(), lag).unwrap();
                let grown = st + lag;
                let expected = if grown >= 300 {
                    None
                } else if can_revive && grown + 60 < 300 && grown + 60 + ttd <= 300 {
                    Some(Profile::Eligible { st: grown, ttd })
                } else {
                    Some(Profile::FailureFatal { st: grown })
                };
                assert_eq!(
                    child, expected,
                    "You checked ST {st}, TTD {ttd}, lag {lag}."
                );
                if let Some(child) = child {
                    assert!(potential(&child) > parent_potential);
                }
            }

            let child = revived_child(&canonical(st, ttd).unwrap()).unwrap();
            let expected = if can_revive {
                revival_count += 1;
                let grown = ttd + st + 60;
                assert!((60..=300).contains(&grown));
                if grown + 60 <= 300 {
                    Some(Profile::Eligible { st: 0, ttd: grown })
                } else {
                    Some(Profile::FailureFatal { st: 0 })
                }
            } else {
                None
            };
            assert_eq!(
                child, expected,
                "You checked revival at ST {st}, TTD {ttd}."
            );
            if let Some(child) = child {
                assert!(potential(&child) > parent_potential);
            }
        }
    }

    assert_eq!(profile_count, 17_011);
    assert_eq!(success_count, 1_020_660);
    assert_eq!(revival_count, 16_711);
}

#[test]
fn e06_catalog_has_the_expected_census_and_reference_ids() {
    let profiles = profile_catalog().unwrap();
    assert_eq!(profiles.len(), 17_011);
    assert_eq!(profiles.len() * profiles.len(), 289_374_121);

    for (id, expected) in [
        (0, Profile::Eligible { st: 0, ttd: 0 }),
        (239, Profile::Eligible { st: 239, ttd: 0 }),
        (240, Profile::Eligible { st: 0, ttd: 60 }),
        (420, Profile::Eligible { st: 180, ttd: 60 }),
        (16_710, Profile::Eligible { st: 0, ttd: 240 }),
        (16_711, Profile::FailureFatal { st: 0 }),
        (16_951, Profile::FailureFatal { st: 240 }),
        (17_010, Profile::FailureFatal { st: 299 }),
    ] {
        assert_eq!(profiles[id], expected, "You assigned profile ID {id}.");
    }
}

#[test]
fn e06_catalog_contains_valid_unique_profiles_in_the_required_order() {
    let profiles = profile_catalog().unwrap();
    assert_eq!(profiles.len(), 17_011);
    let mut counts = [0; 301];
    let mut previous = None;

    for profile in &profiles[..16_711] {
        let Profile::Eligible { st, ttd } = profile else {
            panic!("You inserted a failure-fatal profile before completing the eligible profiles.");
        };
        assert!(*ttd == 0 || (60..=240).contains(ttd));
        assert!(st + 60 < 300 && st + 60 + ttd <= 300);
        counts[*ttd] += 1;

        let key = (*ttd, *st);
        if let Some(prior) = previous {
            assert!(
                prior < key,
                "You repeated a profile or changed the TTD/ST order."
            );
        }
        previous = Some(key);
    }

    for (ttd, count) in counts.into_iter().enumerate() {
        let expected = match ttd {
            0 => 240,
            60..=240 => 241 - ttd,
            _ => 0,
        };
        assert_eq!(count, expected, "You counted profiles at TTD {ttd}.");
    }

    for (st, profile) in profiles[16_711..].iter().enumerate() {
        assert_eq!(profile, &Profile::FailureFatal { st });
    }
}
