use super::*;

fn golden_geometry() -> Geometry {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/cpu-golden.json")).unwrap();
    serde_json::from_value(golden["geometry"].clone()).unwrap()
}
#[test]
fn each_search_geometry_keeps_its_own_projected_mesh() {
    let g = golden_geometry();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/move-search-golden.json")).unwrap();
    let bank: Vec<Fit> = serde_json::from_value(fixture["frames"][1]["bank"].clone()).unwrap();
    assert_eq!(bank.len(), 6);
    let frame = Frame {
        frame: 382,
        width: 720.,
        height: 1280.,
        observations: Observations {
            points: vec![],
            classes: vec![],
            stickers: vec![],
            uncertainties: vec![],
        },
        landmark_presence: vec![],
        colors: vec![],
        pixel_scale: 1.,
    };
    let output = jobs::run(
        &g,
        jobs::Job::Finalize {
            frame: frame.clone(),
            source: None,
            bank: bank.clone(),
            target: None,
            static_mode: false,
        },
    );
    assert_eq!(output["wireframes"].as_array().unwrap().len(), 6);
    assert_eq!(output["wireframe"], output["wireframes"][0]);
    for (i, fit) in bank.iter().enumerate() {
        let alone = jobs::run(
            &g,
            jobs::Job::Finalize {
                frame: frame.clone(),
                source: None,
                bank: vec![fit.clone()],
                target: None,
                static_mode: false,
            },
        );
        assert_eq!(output["wireframes"][i], alone["wireframe"]);
        assert_eq!(output["wireframeFaces"][i], alone["wireframeFaces"][0]);
        assert_eq!(output["wireframeDepths"][i], alone["wireframeDepths"][0]);
        let ids = output["wireframeFaces"][i].as_array().unwrap();
        assert_eq!(ids.len(), output["wireframes"][i].as_array().unwrap().len());
        assert!(ids.iter().all(|id| id.as_u64().unwrap() < 54));
        assert!(output["wireframeDepths"][i]
            .as_array()
            .unwrap()
            .iter()
            .all(|z| z.as_f64().unwrap().is_finite()));
    }
}
#[test]
fn cpu_sticker_groups_and_correspondence_sequences() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/startup-golden.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let input: startup::Input = serde_json::from_value(case["input"].clone()).unwrap();
        let expected: Vec<startup::Match> =
            serde_json::from_value(case["matches"].clone()).unwrap();
        let actual = startup::matches(&input);
        assert_eq!(
            actual.len(),
            expected.len(),
            "{} {}",
            case["name"],
            case["frame"]
        );
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.indices, b.indices);
            assert_eq!(a.sticker_index, b.sticker_index);
            assert_eq!(a.sticker_type, b.sticker_type);
            assert_eq!(a.sequences, b.sequences);
            assert!((a.center_error - b.center_error).abs() < 1e-8);
            assert_eq!(a.mean_presence, b.mean_presence);
        }
    }
}
#[test]
fn fixed_input_sparse_depth_solver_matches_cpu() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/depth-recovery.json")).unwrap();
    let geometry = golden_geometry();
    let workload: Workload = serde_json::from_value(fixture["workload"].clone()).unwrap();
    let seed: [f64; 8] = serde_json::from_value(fixture["seed"].clone()).unwrap();
    let expected: [f64; 8] = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let actual = depth::depth_refine(
        &geometry,
        &workload,
        fixture["axis"].as_u64().unwrap() as usize,
        seed,
        fixture["target"].as_f64().unwrap(),
    );
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() < 1e-6, "parameter {i}: {a} vs {b}");
    }
}
#[test]
fn fixed_input_orientation_preserves_cpu_assignments_and_order() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/orientation.json")).unwrap();
    let g = golden_geometry();
    let frame: Frame = serde_json::from_value(fixture["frame"].clone()).unwrap();
    let bank: Vec<Fit> = serde_json::from_value(fixture["input_bank"].clone()).unwrap();
    let expected: Vec<Fit> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let centers: Vec<Center> = serde_json::from_value(fixture["centers"].clone()).unwrap();
    let x: Vec<f64> = serde_json::from_value(fixture["reference"]["params"].clone()).unwrap();
    let reference = Some((
        rodrigues([x[0], x[1], x[2]]).0,
        fixture["reference"]["frame"].as_u64().unwrap() as usize,
    ));
    let w = work(&frame, &pose(&bank[0]), None, vec![], vec![], false);
    let actual = orientation::orient(
        &g,
        &w,
        bank,
        &centers,
        reference,
        fixture["fps"].as_f64().unwrap(),
        &frame.colors,
    );
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(&expected) {
        assert_eq!(a.nearest, b.nearest);
        assert_eq!(a.valid, b.valid);
        assert_eq!(a.flat_fallback, b.flat_fallback);
        assert!((a.score - b.score).abs() < 1e-6);
    }
}

#[test]
fn stage_entry_rejects_mismatched_checkpoints_and_frame_order() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/orientation.json")).unwrap();
    let frame: Frame = serde_json::from_value(fixture["frame"].clone()).unwrap();
    let g = golden_geometry();
    let mut input = Input {
        frames: vec![frame.clone()],
        fps: 30.,
        turning_start_frame: None,
        expected_center_colors: vec![],
        capture_stages: false,
    };
    let error = run_stage(
        &g,
        StageRequest {
            stage: Stage::Neighbors,
            input: input.clone(),
            state: State::default(),
        },
    )
    .err()
    .unwrap();
    assert!(error.contains("Geometry checkpoint"));
    input.frames.push(frame);
    let error = run_stage(
        &g,
        StageRequest {
            stage: Stage::Startup,
            input,
            state: State::default(),
        },
    )
    .err()
    .unwrap();
    assert!(error.contains("increase strictly"));
}
