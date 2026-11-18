use super::{
    cube::Tables,
    routes::{self, Shape},
    Config, Frame,
};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Geometry {
    pub index: usize,
    pub shape: Shape,
    pub signature: Vec<usize>,
    pub score: f64,
    pub inliers: usize,
    pub p90: f64,
    pub layer_inliers: usize,
    pub observations: Vec<(usize, [f64; 6])>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Row {
    pub before: Option<usize>,
    pub after: usize,
    pub moves: Vec<String>,
    pub sources: Vec<usize>,
    pub degrees: f64,
    pub budget: f64,
}
pub struct Prepared {
    pub frame: usize,
    pub held: bool,
    pub geometries: Vec<Geometry>,
    pub shapes: BTreeMap<usize, Shape>,
    pub previous: Option<usize>,
    pub limit: f64,
    pub rows: Vec<Row>,
    pub ready: bool,
}
pub const COLORS: [&str; 6] = ["blue", "green", "orange", "red", "white", "yellow"];
fn balance(t: &Tables, k: f64, s: f64) -> f64 {
    0.5 * (k / t.keypoint_scale).clamp(0., 1.) + 0.5 * (s / t.sticker_scale).clamp(0., 1.)
}
pub fn compile(t: &Tables, c: &Config, input: Frame) -> Result<Prepared, String> {
    if input.bank.len() > 6 {
        return Err("A frame may contain at most five fits and one flat fallback".into());
    }
    if input.bank.iter().any(|f| {
        f.nearest.len() != input.colors.len()
            || f.valid.len() != input.colors.len()
            || f.nearest.iter().any(|&i| i >= 54)
    }) {
        return Err("Each fit needs one valid face assignment per sticker observation".into());
    }
    if input.colors.iter().any(|s| !COLORS.contains(&s.as_str())) {
        return Err("Unknown sticker color".into());
    }
    let held = c.turning_start_frame.is_some_and(|end| input.frame <= end) || input.bank.is_empty();
    let mut geometries: Vec<Geometry> = vec![];
    let mut ordinary = 0;
    let mut flat = false;
    if !held {
        for (index, f) in input.bank.iter().enumerate() {
            if f.flat_fallback {
                if flat {
                    continue;
                }
                flat = true;
            } else {
                if ordinary >= 5 {
                    continue;
                }
                ordinary += 1;
            }
            if f.angles.iter().any(|&a| a != 0.) && f.metrics.layer_inlier_count == 0 {
                continue;
            }
            let score = balance(t, f.metrics.keypoint_error, f.metrics.sticker_error);
            if !score.is_finite() {
                continue;
            }
            let shape = Shape::new(f.axis, f.angles);
            let signature: Vec<_> = f
                .nearest
                .iter()
                .zip(&f.valid)
                .map(|(&i, &v)| if v { i } else { 255 })
                .collect();
            let mut costs: BTreeMap<usize, [f64; 6]> = BTreeMap::new();
            for ((&face, &valid), predicted) in f.nearest.iter().zip(&f.valid).zip(&input.colors) {
                if valid {
                    let row = costs.entry(face).or_insert([1.; 6]);
                    for (i, truth) in COLORS.iter().enumerate() {
                        let cost = if *truth == predicted {
                            0.
                        } else if t.confusion[*truth].contains(predicted) {
                            0.5
                        } else {
                            1.
                        };
                        row[i] = row[i].min(cost);
                    }
                }
            }
            let g = Geometry {
                index,
                shape,
                signature,
                score,
                inliers: f.metrics.inlier_count,
                p90: f.metrics.p90_error,
                layer_inliers: if f.angles.iter().any(|&a| a != 0.) {
                    f.metrics.layer_inlier_count
                } else {
                    0
                },
                observations: costs.into_iter().collect(),
            };
            // Match _rotation_plans: collapse only identical shape AND assignment.
            if let Some(i) = geometries
                .iter()
                .position(|p| p.shape == g.shape && p.signature == g.signature)
            {
                let p = &geometries[i];
                if (g.score, -(g.inliers as i64), g.p90) < (p.score, -(p.inliers as i64), p.p90) {
                    geometries[i] = g;
                }
            } else {
                geometries.push(g);
            }
        }
    }
    geometries.sort_by_key(|g| g.index);
    let shapes = geometries
        .iter()
        .map(|g| (g.index, g.shape.clone()))
        .collect();
    Ok(Prepared {
        frame: input.frame,
        held,
        geometries,
        shapes,
        previous: None,
        limit: 0.,
        rows: vec![],
        ready: held,
    })
}
fn rows(t: &Tables, frame: &Prepared, previous: Option<&Prepared>) -> Vec<Row> {
    let before = previous
        .map(|p| {
            p.shapes
                .iter()
                .map(|(&i, s)| (Some(i), s.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![(None, Shape::flat())]);
    let mut rows: Vec<Row> = vec![];
    for (&after, shape) in &frame.shapes {
        for (before_index, old) in &before {
            for route in routes::between(old, shape, frame.limit) {
                let (sources, moves) = t.permutation(&route.groups);
                if let Some(row) = rows
                    .iter_mut()
                    .find(|r| r.after == after && r.before == *before_index && r.moves == moves)
                {
                    row.degrees = row.degrees.min(route.degrees);
                    row.budget = route.budget;
                } else {
                    rows.push(Row {
                        before: *before_index,
                        after,
                        moves,
                        sources,
                        degrees: route.degrees,
                        budget: route.budget,
                    });
                }
            }
        }
    }
    rows
}
/// Equal 54/54 limits need no future double-turn classification. Each bank
/// becomes searchable as soon as its preceding eligible bank is linked.
pub fn release(t: &Tables, frames: &mut [Prepared]) -> usize {
    for index in 0..frames.len() {
        if !frames[index].ready {
            let previous = frames[index].previous.map(|j| &frames[j]);
            let prepared = rows(t, &frames[index], previous);
            frames[index].rows = prepared;
            frames[index].ready = true;
        }
    }
    frames.len()
}
