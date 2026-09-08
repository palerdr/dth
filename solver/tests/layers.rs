use solver::profiles::{Profile, potential, profile_catalog};
use solver::sweep::{build_buckets, layer_pairs, layer_sizes};

fn grouped_profiles() -> [Profile; 4] {
    [
        Profile::Eligible { st: 1, ttd: 60 },
        Profile::Eligible { st: 0, ttd: 60 },
        Profile::Eligible { st: 61, ttd: 0 },
        Profile::Eligible { st: 60, ttd: 0 },
    ]
}

#[test]
fn e07_buckets_cover_the_catalog_once_in_ascending_id_order() {
    let profiles = profile_catalog().unwrap();
    let buckets = build_buckets(&profiles);
    let mut seen = vec![false; profiles.len()];

    assert_eq!(buckets[0], [0]);
    assert_eq!(buckets[60], [60, 240]);
    assert_eq!(buckets[301], [16_711]);
    assert_eq!(buckets[600], [17_010]);
    assert!(buckets[241..=300].iter().all(Vec::is_empty));

    for (a, bucket) in buckets.iter().enumerate() {
        assert!(bucket.windows(2).all(|ids| ids[0] < ids[1]));
        for &id in bucket {
            assert!(id < profiles.len());
            assert!(!seen[id], "You stored profile ID {id} more than once.");
            seen[id] = true;
            assert_eq!(potential(&profiles[id]), a);
        }
    }
    assert!(seen.into_iter().all(|present| present));
    assert_eq!(buckets.iter().map(Vec::len).sum::<usize>(), 17_011);
}

#[test]
fn e07_buckets_preserve_ids_when_catalog_potentials_are_out_of_order() {
    let buckets = build_buckets(&grouped_profiles());
    assert_eq!(buckets[60], [1, 3]);
    assert_eq!(buckets[61], [0, 2]);
    assert_eq!(buckets.iter().map(Vec::len).sum::<usize>(), 4);
}

#[test]
fn e07_empty_catalog_has_empty_buckets() {
    assert!(build_buckets(&[]).iter().all(Vec::is_empty));
}

#[test]
fn e07_layer_sizes_match_the_full_census_and_peak() {
    let profiles = profile_catalog().unwrap();
    let buckets = build_buckets(&profiles);
    let sizes = layer_sizes(&buckets);
    assert_eq!(sizes.iter().sum::<usize>(), 289_374_121);
    assert_eq!(sizes[0], 1);
    assert_eq!(sizes[1200], 1);
    assert_eq!(
        sizes
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|&(_, count)| count),
        Some((374, 1_678_715))
    );
}

#[test]
fn e07_layer_sizes_match_a_direct_count_for_a_small_catalog() {
    let profiles = grouped_profiles();
    let buckets = build_buckets(&profiles);
    let sizes = layer_sizes(&buckets);
    let mut expected = [0; 1201];
    for checker in &profiles {
        for dropper in &profiles {
            expected[potential(checker) + potential(dropper)] += 1;
        }
    }
    for p in 0..=1200 {
        assert_eq!(sizes[p], expected[p], "You counted layer {p}.");
    }
    assert_eq!(&sizes[120..=122], &[4, 8, 4]);
}

#[test]
fn e07_empty_catalog_has_zero_layer_sizes() {
    let buckets = build_buckets(&[]);
    assert_eq!(layer_sizes(&buckets), [0; 1201]);
}

#[test]
fn e07_pairs_follow_potential_then_dropper_then_checker_order() {
    let profiles = grouped_profiles();
    let buckets = build_buckets(&profiles);
    assert_eq!(
        layer_pairs(&buckets, 121),
        [
            (1, 0),
            (3, 0),
            (1, 2),
            (3, 2),
            (0, 1),
            (2, 1),
            (0, 3),
            (2, 3)
        ]
    );
    assert_eq!(layer_pairs(&buckets, 120), [(1, 1), (3, 1), (1, 3), (3, 3)]);
}

#[test]
fn e07_pairs_include_both_role_assignments_and_skip_missing_buckets() {
    let profiles = [
        Profile::Eligible { st: 0, ttd: 0 },
        Profile::Eligible { st: 1, ttd: 0 },
        Profile::FailureFatal { st: 0 },
        Profile::FailureFatal { st: 1 },
    ];
    let buckets = build_buckets(&profiles);
    assert_eq!(layer_pairs(&buckets, 302), [(0, 3), (1, 2), (2, 1), (3, 0)]);
    assert!(layer_pairs(&buckets, 10).is_empty());
}

#[test]
fn e07_pairs_match_direct_search_without_duplicates_across_all_layers() {
    let profiles = [
        Profile::Eligible { st: 1, ttd: 60 },
        Profile::Eligible { st: 0, ttd: 0 },
        Profile::FailureFatal { st: 299 },
        Profile::Eligible { st: 0, ttd: 60 },
        Profile::Eligible { st: 61, ttd: 0 },
        Profile::FailureFatal { st: 0 },
        Profile::Eligible { st: 60, ttd: 0 },
        Profile::Eligible { st: 1, ttd: 0 },
    ];
    let buckets = build_buckets(&profiles);
    let mut pair_count = 0;
    for p in 0..=1200 {
        let mut expected = Vec::new();
        for (pc, checker) in profiles.iter().enumerate() {
            for (pd, dropper) in profiles.iter().enumerate() {
                if potential(checker) + potential(dropper) == p {
                    expected.push((pc, pd));
                }
            }
        }
        let mut actual = layer_pairs(&buckets, p);
        pair_count += actual.len();
        actual.sort_unstable();
        assert_eq!(actual, expected, "You enumerated layer {p}.");
    }
    assert_eq!(pair_count, profiles.len() * profiles.len());
}

#[test]
fn e07_pairs_return_empty_outside_the_layer_range() {
    let profiles = grouped_profiles();
    let buckets = build_buckets(&profiles);
    let empty_buckets = build_buckets(&[]);
    for p in [1201, usize::MAX] {
        assert!(layer_pairs(&buckets, p).is_empty());
        assert!(layer_pairs(&empty_buckets, p).is_empty());
    }
}

#[test]
fn e07_empty_catalog_has_no_pairs_at_valid_layers() {
    let buckets = build_buckets(&[]);
    for p in [0, 600, 1200] {
        assert!(layer_pairs(&buckets, p).is_empty());
    }
}

#[test]
fn e07_full_catalog_pair_endpoints_have_one_state() {
    let profiles = profile_catalog().unwrap();
    let buckets = build_buckets(&profiles);
    assert_eq!(layer_pairs(&buckets, 0), [(0, 0)]);
    assert_eq!(layer_pairs(&buckets, 1200), [(17_010, 17_010)]);
}
