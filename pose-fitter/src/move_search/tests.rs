use super::*;
use crate::move_search::routes::{between, Shape};
#[test]
fn immutable_tables_follow_cube_conventions() {
    let t = cube::Tables::load();
    let colors = t.colors("").unwrap();
    assert!(t.solved(&t.identity, &colors));
    let scrambled = t.colors("R U R' U'").unwrap();
    assert!(!t.solved(&t.identity, &scrambled));
    let mut state = t.identity.clone();
    for token in ["U", "R", "U'", "R'"] {
        state = t.moves[token].iter().map(|&i| state[i]).collect();
    }
    assert!(t.solved(&state, &scrambled));
    assert_eq!(t.colors("R2'").unwrap(), t.colors("R2").unwrap());
    assert_eq!(t.colors("r U’").unwrap(), t.colors("Rw U'").unwrap());
}
#[test]
fn shared_rate_has_no_double_exception() {
    let flat = Shape::flat();
    let routes = between(&flat, &flat, 120.);
    assert_eq!(routes.len(), 25);
    assert!(routes.iter().all(|r| r.degrees == 0. || r.degrees == 90.));
    assert!(between(&flat, &flat, 180.).iter().any(|r| r.groups
        == vec![cube::Group {
            axis: 0,
            offsets: [2, 0, 0]
        }]));
    let a = Shape::new(0, [0., 0., 40.]);
    let b = Shape::new(0, [0., 0., -40.]);
    assert!(between(&a, &b, 54.).iter().any(|r| r.groups
        == vec![cube::Group {
            axis: 0,
            offsets: [0, 0, 1]
        }]
        && r.degrees == 10.));
}
#[test]
fn frame_frontier_waits_and_end_is_explicit() {
    let mut engine = Engine::new(Config {
        scramble: "R U".into(),
        ..Config::default()
    })
    .unwrap();
    engine.advance(10);
    assert_eq!(engine.snapshot()["status"], "waiting_for_frames");
    engine
        .push(Frame {
            sequence: 1,
            frame: 2,
            bank: vec![],
            colors: vec![],
        })
        .unwrap();
    engine.advance(10);
    assert_eq!(engine.snapshot()["frames_prepared"], 1);
    assert_eq!(engine.snapshot()["frames_searchable"], 0);
    assert!(engine.end().is_err());
    engine
        .push(Frame {
            sequence: 0,
            frame: 0,
            bank: vec![],
            colors: vec![],
        })
        .unwrap();
    engine.advance(10);
    assert_eq!(engine.snapshot()["frames_searchable"], 2);
    assert_eq!(engine.snapshot()["status"], "waiting_for_frames");
    assert_eq!(engine.snapshot()["stats"]["evaluated_states"], 2);
    engine.end().unwrap();
    engine.advance(10);
    assert_eq!(engine.snapshot()["status"], "exhausted");
    assert!(engine
        .push(Frame {
            sequence: 2,
            frame: 4,
            bank: vec![],
            colors: vec![]
        })
        .is_err());
}
#[test]
fn saved_cpu_scorer_and_widening_oracle() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/move-search-golden.json")).unwrap();
    let mut engine = Engine::new(serde_json::from_value(data["config"].clone()).unwrap()).unwrap();
    for input in data["frames"].as_array().unwrap() {
        engine
            .push(serde_json::from_value(input.clone()).unwrap())
            .unwrap();
    }
    engine.end().unwrap();
    for probe in data["probes"].as_array().unwrap() {
        let actual = engine
            .probe(
                probe["sequence"].as_u64().unwrap() as usize,
                serde_json::from_value(probe["state"].clone()).unwrap(),
                serde_json::from_value(probe["cue"].clone()).unwrap(),
            )
            .unwrap();
        let expected = probe["choices"].as_array().unwrap();
        let actual = actual["choices"].as_array().unwrap();
        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(actual) {
            for key in [
                "geometry_option_index",
                "previous_geometry_index",
                "state_moves",
                "color_match_count",
                "color_forbidden_count",
            ] {
                assert_eq!(a[key], b[key], "{key}");
            }
            for key in [
                "state_score",
                "color_error",
                "angular_travel_degrees",
                "rotation_budget_degrees",
            ] {
                if let Some(value) = a[key].as_f64() {
                    assert!((value - b[key].as_f64().unwrap()).abs() < 1e-12, "{key}");
                }
            }
        }
    }
    engine.advance(10000);
    let actual = engine.snapshot();
    assert_eq!(actual["status"], data["expected"]["status"]);
    for (key, value) in data["expected"]["stats"].as_object().unwrap() {
        assert_eq!(actual["stats"][key], *value, "{key}");
    }
}
