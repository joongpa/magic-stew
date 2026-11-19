//! Coupled endpoint motion in camera space. No fitting or color scoring.
//!
//! A rigid turn and a layer turn combine before rate checks. Measuring material
//! cubies also removes the fitter's middle-slab gauge: an M turn must not be
//! mistaken for a whole-cube rotation just because its middle angle is in R.
use super::{cube::Group, routes::Shape};
use crate::math::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Motion {
    pub rigid_degrees: f64,
    pub combined_degrees: f64,
}
fn turn(axis: usize, degrees: f64) -> M3 {
    let (s, c) = degrees.to_radians().sin_cos();
    let mut result = identity();
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    result[a][a] = c;
    result[b][b] = c;
    result[a][b] = -s;
    result[b][a] = s;
    result
}
fn slab(shape: &Shape, position: [i32; 3]) -> M3 {
    shape
        .axis
        .map(|axis| turn(axis, shape.angles[(position[axis] + 1) as usize]))
        .unwrap_or_else(identity)
}
pub fn prepare(old: &Shape, new: &Shape, groups: &[Group]) -> Vec<M3> {
    let mut result = vec![];
    let mut include = |delta: M3| {
        if !result.contains(&delta) {
            result.push(delta);
        }
    };
    let axis = old
        .axis
        .or(new.axis)
        .or_else(|| groups.first().map(|g| g.axis));
    if let Some(axis) = axis.filter(|&a| {
        old.axis.is_none_or(|v| v == a)
            && new.axis.is_none_or(|v| v == a)
            && groups.iter().all(|g| g.axis == a)
    }) {
        for layer in 0..3 {
            let degrees = new.angles[layer] - old.angles[layer]
                + 90. * groups.iter().map(|g| g.offsets[layer]).sum::<i32>() as f64;
            include(turn(axis, degrees));
        }
    } else if groups.is_empty() && old.axis.is_none() && new.axis.is_none() {
        include(identity());
    } else {
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let start = [x, y, z];
                    let mut position = start;
                    let mut moved = identity();
                    for group in groups {
                        let q = turn(
                            group.axis,
                            90. * group.offsets[(position[group.axis] + 1) as usize] as f64,
                        );
                        moved = mm(q, moved);
                        position = mv(q, position.map(|v| v as f64)).map(|v| v.round() as i32);
                    }
                    include(mm(
                        mm(slab(new, position), moved),
                        transpose(slab(old, start)),
                    ));
                }
            }
        }
    }
    result
}
pub fn measure_prepared(before: M3, after: M3, material: &[M3]) -> Motion {
    let camera = mm(transpose(before), after);
    let (mut low, mut high) = (1_f64, -1_f64);
    for delta in material {
        let trace = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| camera[i][j] * delta[j][i])
            .sum::<f64>();
        let cosine = ((trace - 1.) * 0.5).clamp(-1., 1.);
        low = low.min(cosine);
        high = high.max(cosine);
    }
    Motion {
        rigid_degrees: high.acos().to_degrees(),
        combined_degrees: low.acos().to_degrees(),
    }
}
pub fn measure(before: M3, old: &Shape, after: M3, new: &Shape, groups: &[Group]) -> Motion {
    measure_prepared(before, after, &prepare(old, new, groups))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_and_turn_combine_before_limits() {
        // R x' = Lw in the immutable renderer convention.
        let flat = Shape::flat();
        let r = Group {
            axis: 0,
            offsets: [0, 0, -1],
        };
        let combined = measure(identity(), &flat, axis_rotation(0, 90.), &flat, &[r]);
        assert!(combined.rigid_degrees < 1e-6);
        assert!((combined.combined_degrees - 90.).abs() < 1e-6);
        let hold = measure(identity(), &flat, axis_rotation(0, 90.), &flat, &[]);
        assert!((hold.rigid_degrees - 90.).abs() < 1e-6);
        let wide = measure(
            identity(),
            &flat,
            identity(),
            &flat,
            &[Group {
                axis: 0,
                offsets: [1, 1, 0],
            }],
        );
        assert!((wide.rigid_degrees - combined.rigid_degrees).abs() < 1e-6);
        assert!((wide.combined_degrees - combined.combined_degrees).abs() < 1e-6);
        let tables = super::super::cube::Tables::load();
        let mut state = tables.identity.clone();
        for token in ["R", "x'"] {
            state = tables.moves[token].iter().map(|&i| state[i]).collect();
        }
        let expected: Vec<_> = tables.moves["Lw"]
            .iter()
            .map(|&i| tables.identity[i])
            .collect();
        assert_eq!(state, expected);
    }
    #[test]
    fn middle_slab_gauge_is_layer_motion() {
        let moved = Shape::new(0, [-45., 0., -45.]);
        let m = measure(
            identity(),
            &Shape::flat(),
            axis_rotation(0, 45.),
            &moved,
            &[],
        );
        assert!(m.rigid_degrees < 1e-6);
        assert!((m.combined_degrees - 45.).abs() < 1e-6);
    }
    #[test]
    fn unrelated_rotation_increases_total_motion() {
        let m = measure(
            identity(),
            &Shape::flat(),
            axis_rotation(1, 45.),
            &Shape::flat(),
            &[Group {
                axis: 0,
                offsets: [0, 0, 1],
            }],
        );
        assert!((m.rigid_degrees - 45.).abs() < 1e-6);
        assert!(m.combined_degrees > 90.);
    }
}
