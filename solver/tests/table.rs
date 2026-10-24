use solver::profiles::{Profile, profile_catalog, profile_id};
use solver::stage::ACTIONS;
use solver::table::{Table, Transitions};

#[test]
fn profile_id_matches_the_catalog_position() {
    let catalog = profile_catalog().unwrap();
    for (id, profile) in catalog.iter().enumerate() {
        assert_eq!(
            profile_id(profile) as usize,
            id,
            "You assigned {profile:?} ID {id}."
        );
    }
}

#[test]
fn table_reserves_the_win_column_and_leaves_classes_nan() {
    let table = Table::new(3).unwrap();
    assert_eq!(table.values.len(), 12);
    for pd in 0..3 {
        for pc in 0..3 {
            assert!(table.get(pc, pd).is_nan());
        }
        assert_eq!(table.values[table.index(pd, 3)], -1.0);
    }
}

#[test]
fn gather_reads_the_dropper_row_through_the_checker_transitions() {
    let n = 3;
    let mut table = Table::new(n).unwrap();
    for pc in 0..n {
        for pd in 0..n {
            let idx = table.index(pc, pd);
            table.values[idx] = (10 * pc + pd) as f64 / 100.0;
        }
    }
    let mut succ = vec![[n as u32; ACTIONS]; n];
    succ[1][0] = 0;
    succ[1][1] = 2;
    let t = Transitions {
        profiles: vec![
            Profile::FailureFatal { st: 0 },
            Profile::FailureFatal { st: 1 },
            Profile::FailureFatal { st: 2 },
        ],
        succ,
        fail: vec![n as u32, 0, n as u32],
        rev: vec![0.0, 0.25, 0.0],
    };
    let stage = table.gather(&t, 1, 2).unwrap();
    let m = stage.matrix();
    assert_eq!(m[0], -0.20, "S[0] reads V[pd=2, succ=0].");
    assert_eq!(m[1], -0.22, "S[1] reads V[pd=2, succ=2].");
    assert_eq!(m[2], 1.0, "A terminal success pays the Dropper +1.");
    assert_eq!(
        m[ACTIONS],
        0.25 * -0.20 + 0.75,
        "F mixes the revival child with 1 - p."
    );
    let fatal = table.gather(&t, 0, 1).unwrap().matrix();
    assert_eq!(fatal[ACTIONS], 1.0, "Absent revival pays the Dropper +1.");
}

#[test]
fn endgame_matches_the_closed_failure_fatal_values() {
    let t = Transitions::build();
    let mut table = Table::new(t.len()).unwrap();
    table.sweep(&t, 1194).unwrap();
    let ff = |st: usize| profile_id(&Profile::FailureFatal { st }) as usize;
    for st in 296..=299 {
        assert_eq!(table.get(ff(299), ff(st)), 1.0);
    }
    assert!((table.get(ff(298), ff(299)) - 29.0 / 30.0).abs() < 1e-12);
    assert!(
        table.get(ff(292), ff(299)).is_nan(),
        "Layers below stop_at stay NaN."
    );
}
