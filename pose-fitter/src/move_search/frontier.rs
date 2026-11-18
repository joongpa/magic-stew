use super::{
    cube::Tables,
    prepared::{Geometry, Prepared, COLORS},
    Config,
};
use serde::Serialize;
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
};

#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub frame_index: usize,
    pub geometry_option_index: Option<usize>,
    pub previous_geometry_index: Option<usize>,
    pub state_moves: Vec<String>,
    pub state_score: f64,
    pub color_error: f64,
    pub color_match_count: usize,
    pub color_correct_count: usize,
    pub color_confusion_count: usize,
    pub color_forbidden_count: usize,
    pub angular_travel_degrees: f64,
    pub rotation_budget_degrees: f64,
    pub rotation_budget: bool,
    pub sibling_rank: usize,
    pub branch_cutoff_bypassed: bool,
    pub queue_priority: f64,
    pub window_mismatch_rate: Option<f64>,
    pub window_mismatch_penalty: f64,
}
impl Step {
    fn hold(frame: usize, cue: Option<usize>) -> Self {
        Self {
            frame_index: frame,
            geometry_option_index: cue,
            previous_geometry_index: cue,
            state_moves: vec![],
            state_score: 0.,
            color_error: 0.,
            color_match_count: 0,
            color_correct_count: 0,
            color_confusion_count: 0,
            color_forbidden_count: 0,
            angular_travel_degrees: 0.,
            rotation_budget_degrees: 0.,
            rotation_budget: true,
            sibling_rank: 1,
            branch_cutoff_bypassed: false,
            queue_priority: 0.,
            window_mismatch_rate: None,
            window_mismatch_penalty: 0.,
        }
    }
}
#[derive(Clone)]
pub struct Choice {
    pub state: Vec<u8>,
    pub cue: Option<usize>,
    pub step: Step,
    pub rank: [f64; 4],
}
fn color(t: &Tables, g: &Geometry, state: &[u8], colors: &[usize]) -> (f64, usize, usize, usize) {
    let (mut correct, mut confused, mut forbidden) = (0, 0, 0);
    for (face, costs) in &g.observations {
        let expected = colors[state[t.geometry_positions[*face]] as usize];
        if costs[expected] == 0. {
            correct += 1;
        } else if costs[expected] < 1. {
            confused += 1;
        } else {
            forbidden += 1;
        }
    }
    let total = correct + confused + forbidden;
    let error = if total > 0 {
        (0.5 * confused as f64 + forbidden as f64) / total as f64
    } else if g.signature.is_empty() {
        0.
    } else {
        f64::INFINITY
    };
    (error, correct, confused, forbidden)
}
pub fn choices(
    t: &Tables,
    p: &Prepared,
    state: &[u8],
    cue: Option<usize>,
    colors: &[usize],
) -> Vec<Choice> {
    if p.held {
        return vec![Choice {
            state: state.to_vec(),
            cue,
            step: Step::hold(p.frame, cue),
            rank: [0.; 4],
        }];
    }
    let mut result: Vec<Choice> = vec![];
    let mut seen = HashSet::new();
    for row in p.rows.iter().filter(|r| r.before == cue) {
        // Identical endpoint geometry and permutation have identical color and
        // geometry ranks. Retain the first row, just as the CPU scorer does.
        if !seen.insert((row.after, row.sources.as_slice())) {
            continue;
        }
        let g = p.geometries.iter().find(|g| g.index == row.after).unwrap();
        let after: Vec<_> = row.sources.iter().map(|&i| state[i]).collect();
        let (error, correct, confused, forbidden) = color(t, g, &after, colors);
        let score = 0.5 * g.score + 0.5 * error;
        if !score.is_finite() {
            continue;
        }
        let rank = [score, -(g.inliers as f64), g.p90, -(g.layer_inliers as f64)];
        let step = Step {
            frame_index: p.frame,
            geometry_option_index: Some(row.after),
            previous_geometry_index: row.before,
            state_moves: row.moves.clone(),
            state_score: score,
            color_error: error,
            color_match_count: correct + confused + forbidden,
            color_correct_count: correct,
            color_confusion_count: confused,
            color_forbidden_count: forbidden,
            angular_travel_degrees: row.degrees,
            rotation_budget_degrees: row.budget,
            rotation_budget: true,
            sibling_rank: 0,
            branch_cutoff_bypassed: false,
            queue_priority: 0.,
            window_mismatch_rate: None,
            window_mismatch_penalty: 0.,
        };
        let choice = Choice {
            state: after,
            cue: Some(row.after),
            step,
            rank,
        };
        result.push(choice);
    }
    result.sort_by(|a, b| rank_cmp(&a.rank, &b.rank));
    result
}
fn rank_cmp(a: &[f64; 4], b: &[f64; 4]) -> Ordering {
    for i in 0..4 {
        let order = a[i].partial_cmp(&b[i]).unwrap();
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct Context {
    depth: usize,
    state: Vec<u8>,
    cue: Option<usize>,
}
#[derive(Clone)]
struct Node {
    context: Context,
    priority: f64,
    parent: Option<usize>,
    step: Option<Step>,
    window: Vec<(usize, usize)>,
    status: u8,
}
#[derive(Clone, Copy)]
struct Entry {
    priority: f64,
    depth: usize,
    serial: usize,
    id: usize,
}
impl PartialEq for Entry {
    fn eq(&self, b: &Self) -> bool {
        self.serial == b.serial
    }
}
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, b: &Self) -> Ordering {
        b.priority
            .partial_cmp(&self.priority)
            .unwrap()
            .then(self.depth.cmp(&b.depth))
            .then(b.serial.cmp(&self.serial))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
#[derive(Default, Serialize)]
pub struct Stats {
    pub evaluated_states: usize,
    pub attempted_transitions: usize,
    pub dominated_pruned: usize,
    pub reopened_states: usize,
    pub improved_pending: usize,
    pub deferred_promoted: usize,
    pub furthest_frames: usize,
    pub unsolved_endpoints: usize,
    pub branch_stage: usize,
    pub peak_pending: usize,
}
pub struct Frontier {
    pub stats: Stats,
    nodes: Vec<Node>,
    pending: BinaryHeap<Entry>,
    blocked: Vec<usize>,
    seen: HashMap<Context, usize>,
    deferred: HashMap<(usize, Context), usize>,
    serial: usize,
    stage_start: usize,
    pub winner: Option<usize>,
    best: usize,
    pub status: String,
    pub colors: Vec<String>,
    color_ids: Vec<usize>,
}
impl Frontier {
    pub fn add_static(&mut self) {
        if self.stats.branch_stage == 0 {
            self.stage_start += 1;
        }
    }
    pub fn new(t: &Tables, c: &Config) -> Result<Self, String> {
        let colors = t.colors(&c.scramble)?;
        let color_ids = colors
            .iter()
            .map(|s| COLORS.iter().position(|&v| v == s).unwrap())
            .collect();
        let root = Node {
            context: Context {
                depth: 0,
                state: t.identity.clone(),
                cue: None,
            },
            priority: 0.,
            parent: None,
            step: None,
            window: vec![],
            status: 0,
        };
        let mut s = Self {
            stats: Stats::default(),
            nodes: vec![],
            pending: BinaryHeap::new(),
            blocked: vec![],
            seen: HashMap::new(),
            deferred: HashMap::new(),
            serial: 0,
            stage_start: 0,
            winner: None,
            best: 0,
            status: "waiting_for_frames".into(),
            colors,
            color_ids,
        };
        s.offer(root, t, &[], false);
        if t.solved(&t.identity, &s.colors) {
            s.winner = Some(0);
            s.status = "already_solved".into();
        }
        Ok(s)
    }
    fn enqueue(&mut self, id: usize, t: &Tables, frames: &[Prepared], ended: bool) {
        self.serial += 1;
        let n = &mut self.nodes[id];
        if t.solved(&n.context.state, &self.colors)
            || (n.context.depth < frames.len() && frames[n.context.depth].ready)
            || ended && n.context.depth == frames.len()
        {
            n.status = 0;
            self.pending.push(Entry {
                id,
                priority: n.priority,
                depth: n.context.depth,
                serial: self.serial,
            });
        } else {
            n.status = 1;
            self.blocked.push(id);
        }
    }
    fn offer(&mut self, mut node: Node, t: &Tables, frames: &[Prepared], ended: bool) -> bool {
        if let Some(&old) = self.seen.get(&node.context) {
            if self.nodes[old].priority <= node.priority {
                self.stats.dominated_pruned += 1;
                return false;
            }
            if self.nodes[old].status == 2 {
                self.stats.reopened_states += 1;
            } else {
                self.stats.improved_pending += 1;
            }
            self.nodes[old].status = 4;
        }
        let id = self.nodes.len();
        node.status = 0;
        self.seen.insert(node.context.clone(), id);
        self.nodes.push(node);
        if self.nodes[id].context.depth > self.nodes[self.best].context.depth {
            self.best = id;
        }
        self.enqueue(id, t, frames, ended);
        true
    }
    pub fn unblock(&mut self, t: &Tables, frames: &[Prepared], ended: bool) {
        let blocked = std::mem::take(&mut self.blocked);
        for id in blocked {
            if self.nodes[id].status == 1 {
                self.enqueue(id, t, frames, ended);
            }
        }
    }
    fn clean(&mut self) {
        while self
            .pending
            .peek()
            .is_some_and(|e| self.nodes[e.id].status != 0)
        {
            self.pending.pop();
        }
    }
    pub fn run(&mut self, t: &Tables, c: &Config, frames: &[Prepared], ended: bool, budget: usize) {
        if self.winner.is_some() || self.status == "state_limit" {
            return;
        }
        self.unblock(t, frames, ended);
        let limit = self.stats.evaluated_states.saturating_add(budget);
        loop {
            self.clean();
            if self.stats.evaluated_states >= c.state_limit {
                self.status = "state_limit".into();
                break;
            }
            let waiting = self.blocked.iter().any(|&i| self.nodes[i].status == 1);
            let stage = self.stats.branch_stage;
            if stage + 1 < c.branch_widths.len()
                && ((self.pending.is_empty() && !waiting)
                    || self.stats.evaluated_states.saturating_sub(self.stage_start)
                        >= c.branch_stage_expansions)
            {
                self.stats.branch_stage += 1;
                self.stage_start = self.stats.evaluated_states;
                let width = c.branch_widths[self.stats.branch_stage];
                let mut ready: Vec<_> = self
                    .deferred
                    .iter()
                    .filter_map(|((rank, _), &id)| (*rank <= width).then_some((*rank, id)))
                    .collect();
                // Python promotes by required width then deferred queue serial.
                ready.sort_by(|&(ar, a), &(br, b)| {
                    ar.cmp(&br)
                        .then(
                            self.nodes[a]
                                .priority
                                .partial_cmp(&self.nodes[b].priority)
                                .unwrap(),
                        )
                        .then(
                            self.nodes[b]
                                .context
                                .depth
                                .cmp(&self.nodes[a].context.depth),
                        )
                        .then(a.cmp(&b))
                });
                for (rank, id) in ready {
                    let node = self.nodes[id].clone();
                    self.deferred.remove(&(rank, node.context.clone()));
                    if self.offer(node, t, frames, ended) {
                        self.stats.deferred_promoted += 1;
                    }
                }
                continue;
            }
            if self.pending.is_empty() {
                self.status = if waiting || !ended {
                    "waiting_for_frames"
                } else {
                    "exhausted"
                }
                .into();
                break;
            }
            if self.stats.evaluated_states >= limit {
                self.status = "running".into();
                break;
            }
            let entry = self.pending.pop().unwrap();
            let id = entry.id;
            let node = self.nodes[id].clone();
            self.nodes[id].status = 2;
            self.stats.furthest_frames = self.stats.furthest_frames.max(node.context.depth);
            if t.solved(&node.context.state, &self.colors) {
                self.winner = Some(id);
                self.status = "solved".into();
                break;
            }
            if node.context.depth == frames.len() {
                self.stats.unsolved_endpoints += 1;
                continue;
            }
            self.stats.evaluated_states += 1;
            let options = choices(
                t,
                &frames[node.context.depth],
                &node.context.state,
                node.context.cue,
                &self.color_ids,
            );
            let unrestricted = c.unrestricted_sticker_count.is_some_and(|threshold| {
                !options.is_empty()
                    && options
                        .iter()
                        .all(|v| v.step.color_match_count <= threshold)
            });
            let (mut moves_seen, mut contexts) = (HashSet::new(), HashSet::new());
            let mut prior_rank = None;
            let mut admission = 0;
            for mut choice in options {
                self.stats.attempted_transitions += 1;
                let key = Context {
                    depth: node.context.depth + 1,
                    state: choice.state,
                    cue: choice.cue,
                };
                if moves_seen.contains(&choice.step.state_moves) || contexts.contains(&key) {
                    continue;
                }
                moves_seen.insert(choice.step.state_moves.clone());
                contexts.insert(key.clone());
                let rank = contexts.len();
                if prior_rank.is_none_or(|p| rank_cmp(&p, &choice.rank) != Ordering::Equal) {
                    admission = rank;
                    prior_rank = Some(choice.rank);
                }
                if !unrestricted && admission > *c.branch_widths.last().unwrap() {
                    break;
                }
                choice.step.sibling_rank = rank;
                choice.step.branch_cutoff_bypassed = unrestricted;
                let mut window = node.window.clone();
                let total = choice.step.color_match_count;
                if total > 0 || !window.is_empty() {
                    if window.len() == 8 {
                        window.remove(0);
                    }
                    window.push((choice.step.color_forbidden_count, total));
                }
                let (misses, total) = window.iter().fold((0, 0), |(m, n), (a, b)| (m + a, n + b));
                let rate = (total > 0).then(|| misses as f64 / total as f64);
                let penalty = if window.len() == 8 {
                    c.mismatch_window_weight * rate.unwrap_or(0.)
                } else {
                    0.
                };
                choice.step.window_mismatch_rate = rate;
                choice.step.window_mismatch_penalty = penalty;
                let priority = 0.5 * choice.step.color_error + penalty;
                choice.step.queue_priority = priority;
                let child = Node {
                    context: key,
                    priority,
                    parent: Some(id),
                    step: Some(choice.step),
                    window,
                    status: 0,
                };
                if !unrestricted && admission > c.branch_widths[self.stats.branch_stage] {
                    let key = (admission, child.context.clone());
                    if self
                        .deferred
                        .get(&key)
                        .is_none_or(|&old| self.nodes[old].priority > priority)
                    {
                        let child_id = self.nodes.len();
                        self.nodes.push(child);
                        self.deferred.insert(key, child_id);
                        self.serial += 1;
                    }
                } else {
                    self.offer(child, t, frames, ended);
                }
            }
            self.stats.peak_pending = self
                .stats
                .peak_pending
                .max(self.pending.len() + self.blocked.len());
        }
    }
    pub fn path(&self) -> Vec<Step> {
        let mut path = vec![];
        let mut at = Some(self.winner.unwrap_or(self.best));
        while let Some(i) = at {
            let n = &self.nodes[i];
            if let Some(step) = &n.step {
                path.push(step.clone());
            }
            at = n.parent;
        }
        path.reverse();
        path
    }
    pub fn snapshot(
        &self,
        frames: &[Prepared],
        waiting_preparation: usize,
        ended: bool,
    ) -> serde_json::Value {
        let path = self.path();
        let winner = self.winner.map(|i| &self.nodes[i]);
        serde_json::json!({"status":self.status,"stats":self.stats,"frames_received":frames.len()+waiting_preparation,
            "frames_prepared":frames.len()+waiting_preparation,"frames_searchable":frames.iter().take_while(|f|f.ready).count(),
            "pending":self.pending.len(),"blocked":self.blocked.iter().filter(|&&i|self.nodes[i].status==1).count(),
            "deferred":self.deferred.len(),"ended":ended,"steps":path,
            "solved_frame_index":winner.and_then(|n|n.step.as_ref().map(|s|s.frame_index))})
    }
    pub fn probe(&self, t: &Tables, p: &Prepared, state: &[u8], cue: Option<usize>) -> Vec<Step> {
        choices(t, p, state, cue, &self.color_ids)
            .into_iter()
            .map(|c| c.step)
            .collect()
    }
}
