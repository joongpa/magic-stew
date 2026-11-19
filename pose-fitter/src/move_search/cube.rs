use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Group {
    pub axis: usize,
    pub offsets: [i32; 3],
}
#[derive(Deserialize)]
pub struct GroupSpec {
    #[serde(flatten)]
    pub group: Group,
    pub moves: Vec<String>,
    pub sources: Vec<usize>,
}
#[derive(Deserialize)]
pub struct Tables {
    pub identity: Vec<u8>,
    pub geometry_positions: Vec<usize>,
    pub face_groups: Vec<Vec<usize>>,
    pub solved_colors: Vec<String>,
    pub groups: Vec<GroupSpec>,
    pub moves: HashMap<String, Vec<usize>>,
    pub confusion: HashMap<String, Vec<String>>,
    pub keypoint_scale: f64,
    pub sticker_scale: f64,
    #[serde(skip)]
    permutations: RefCell<HashMap<Vec<Group>, (Vec<usize>, Vec<String>)>>,
}
impl Tables {
    pub fn load() -> Self {
        serde_json::from_str(include_str!("../../tables/move-tables.json")).unwrap()
    }
    pub fn colors(&self, scramble: &str) -> Result<Vec<String>, String> {
        let mut colors: Vec<_> = self
            .identity
            .iter()
            .map(|&i| self.solved_colors[i as usize].clone())
            .collect();
        let normalized = scramble.replace('’', "'");
        for token in normalized.split_whitespace() {
            let token = token.trim_end_matches("2'").to_string()
                + if token.ends_with("2'") { "2" } else { "" };
            let sources = self
                .moves
                .get(&token)
                .ok_or_else(|| format!("Unknown scramble move: {token}"))?;
            colors = sources.iter().map(|&i| colors[i].clone()).collect();
        }
        // Colors are keyed by encoded identity label, not by position.
        let mut result = vec![String::new(); 54];
        for (position, &label) in self.identity.iter().enumerate() {
            result[label as usize] = colors[position].clone();
        }
        Ok(result)
    }
    pub fn permutation(&self, groups: &[Group]) -> (Vec<usize>, Vec<String>) {
        if let Some(value) = self.permutations.borrow().get(groups) {
            return value.clone();
        }
        let mut sources: Vec<_> = (0..54).collect();
        let mut moves = vec![];
        for group in groups {
            let spec = self.groups.iter().find(|g| g.group == *group).unwrap();
            sources = spec.sources.iter().map(|&i| sources[i]).collect();
            moves.extend(spec.moves.clone());
        }
        self.permutations
            .borrow_mut()
            .insert(groups.to_vec(), (sources.clone(), moves.clone()));
        (sources, moves)
    }
    pub fn solved(&self, state: &[u8], colors: &[String]) -> bool {
        let first: Vec<_> = self
            .face_groups
            .iter()
            .map(|g| &colors[state[g[0]] as usize])
            .collect();
        self.face_groups
            .iter()
            .zip(&first)
            .all(|(g, c)| g.iter().all(|&p| &colors[state[p] as usize] == *c))
            && (0..6).all(|i| (0..i).all(|j| first[i] != first[j]))
    }
    /// Scramble inversion followed by all 24 rigid orientations. No observed
    /// move sequence or fitted pose is used to construct the solved boundary.
    pub fn solved_seeds(&self, scramble: &str) -> Result<Vec<Vec<u8>>, String> {
        let normalized = scramble.replace('’', "'");
        let mut state = self.identity.clone();
        for token in normalized.split_whitespace().rev() {
            let inverse = if token.ends_with('2') || token.ends_with("2'") {
                token.trim_end_matches('\'').to_string()
            } else if token.ends_with('\'') {
                token.trim_end_matches('\'').to_string()
            } else {
                format!("{token}'")
            };
            let sources = self.moves.get(&inverse).ok_or_else(|| format!("Unknown scramble move: {token}"))?;
            state = sources.iter().map(|&i| state[i]).collect();
        }
        let mut states = vec![state];
        let mut seen = std::collections::HashSet::from([states[0].clone()]);
        let mut index = 0;
        while index < states.len() {
            for name in ["x", "y", "z"] {
                let next: Vec<_> = self.moves[name].iter().map(|&i| states[index][i]).collect();
                if seen.insert(next.clone()) { states.push(next); }
            }
            index += 1;
        }
        Ok(states)
    }
}
pub fn normalize(groups: &[Group]) -> Vec<Group> {
    let mut result: Vec<Group> = vec![];
    for group in groups {
        let mut q = group.offsets;
        if result.last().is_some_and(|p| p.axis == group.axis) {
            let previous = result.pop().unwrap();
            for i in 0..3 {
                q[i] += previous.offsets[i];
            }
        }
        for v in &mut q {
            *v = v.rem_euclid(4);
            if *v > 2 {
                *v -= 4;
            }
        }
        if q != [0; 3] {
            result.push(Group {
                axis: group.axis,
                offsets: q,
            });
        }
    }
    result
}
