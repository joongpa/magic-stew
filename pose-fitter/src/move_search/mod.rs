//! Incremental forward move search. Numerical preparation and search state are
//! independent of a decoder, worker scheduler or UI. No future-frame guesses.
pub mod cube;
mod frontier;
mod prepared;
pub mod routes;
use crate::Fit;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub scramble: String,
    pub fps: f64,
    pub max_tps: f64,
    pub turning_start_frame: Option<usize>,
    pub branch_widths: Vec<usize>,
    pub branch_stage_expansions: usize,
    pub state_limit: usize,
    pub mismatch_window_weight: f64,
    pub unrestricted_sticker_count: Option<usize>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            scramble: String::new(),
            fps: 30.,
            max_tps: 54.,
            turning_start_frame: None,
            branch_widths: vec![3],
            branch_stage_expansions: 10000,
            state_limit: 1_000_000,
            mismatch_window_weight: 1.,
            unrestricted_sticker_count: Some(3),
        }
    }
}
#[derive(Deserialize, Clone)]
pub struct Frame {
    pub sequence: usize,
    pub frame: usize,
    pub bank: Vec<Fit>,
    pub colors: Vec<String>,
}
pub struct Engine {
    tables: cube::Tables,
    config: Config,
    pending: BTreeMap<usize, prepared::Prepared>,
    frames: Vec<prepared::Prepared>,
    frontier: frontier::Frontier,
    last_active: Option<usize>,
    baseline: Option<i64>,
    ended: bool,
}
impl Engine {
    pub fn new(config: Config) -> Result<Self, String> {
        if !config.fps.is_finite()
            || config.fps <= 0.
            || config.max_tps != 54.
            || config.branch_widths.is_empty()
            || config.branch_widths.iter().any(|&v| v == 0)
            || config.branch_widths.windows(2).any(|p| p[0] >= p[1])
            || config.branch_stage_expansions == 0
            || config.state_limit == 0
            || !config.mismatch_window_weight.is_finite()
            || config.mismatch_window_weight < 0.
        {
            return Err("Move search requires positive FPS, ordered branch widths, finite weights and the shared 54 TPS limit".into());
        }
        let tables = cube::Tables::load();
        let frontier = frontier::Frontier::new(&tables, &config)?;
        Ok(Self {
            tables,
            config,
            pending: BTreeMap::new(),
            frames: vec![],
            frontier,
            last_active: None,
            baseline: None,
            ended: false,
        })
    }
    pub fn push(&mut self, input: Frame) -> Result<(), String> {
        if self.ended {
            return Err("Move search input is closed".into());
        }
        let sequence = input.sequence;
        if sequence < self.frames.len() || self.pending.contains_key(&sequence) {
            return Err("Duplicate move-search frame sequence".into());
        }
        let prepared = prepared::compile(&self.tables, &self.config, input)?;
        self.pending.insert(sequence, prepared);
        while let Some(mut p) = self.pending.remove(&self.frames.len()) {
            if self.frames.last().is_some_and(|old| p.frame <= old.frame) {
                return Err("Move-search source frames must increase".into());
            }
            self.baseline.get_or_insert(p.frame as i64 - 1);
            if self
                .config
                .turning_start_frame
                .is_some_and(|end| p.frame <= end)
            {
                self.last_active = None;
                self.baseline = Some(p.frame as i64);
                self.frontier.add_static();
            }
            if !p.held {
                p.previous = self.last_active;
                let previous = self
                    .last_active
                    .map(|i| self.frames[i].frame as i64)
                    .unwrap_or(self.baseline.unwrap());
                p.limit = 90. * self.config.max_tps * (p.frame as f64 - previous as f64)
                    / self.config.fps;
                self.last_active = Some(self.frames.len());
            }
            self.frames.push(p);
        }
        prepared::release(&self.tables, &mut self.frames);
        self.frontier.unblock(&self.tables, &self.frames, false);
        Ok(())
    }
    pub fn end(&mut self) -> Result<(), String> {
        if !self.pending.is_empty() {
            return Err("Cannot close move search with a missing earlier frame".into());
        }
        self.ended = true;
        self.frontier.unblock(&self.tables, &self.frames, true);
        Ok(())
    }
    pub fn advance(&mut self, budget: usize) {
        self.frontier.run(
            &self.tables,
            &self.config,
            &self.frames,
            self.ended,
            budget.min(100_000),
        );
    }
    pub fn snapshot(&self) -> serde_json::Value {
        self.frontier
            .snapshot(&self.frames, self.pending.len(), self.ended)
    }
    pub fn probe(
        &self,
        sequence: usize,
        state: Vec<u8>,
        cue: Option<usize>,
    ) -> Result<serde_json::Value, String> {
        if state.len() != 54 || state.iter().any(|&v| v >= 54) {
            return Err("Search state must have 54 encoded identities".into());
        }
        let p = self.frames.get(sequence).ok_or("Frame is not ready")?;
        Ok(
            serde_json::json!({"rows":p.rows,"choices":self.frontier.probe(&self.tables,p,&state,cue)}),
        )
    }
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Init {
        config: Config,
    },
    Frame {
        frame: Frame,
    },
    Advance {
        budget: usize,
    },
    End,
    Status,
    Probe {
        sequence: usize,
        state: Vec<u8>,
        cue: Option<usize>,
    },
}
pub fn request(slot: &mut Option<Engine>, request: Request) -> Result<serde_json::Value, String> {
    if let Request::Init { config } = request {
        *slot = Some(Engine::new(config)?);
        return Ok(slot.as_ref().unwrap().snapshot());
    }
    let engine = slot
        .as_mut()
        .ok_or("Move search has not been initialized")?;
    match request {
        Request::Frame { frame } => engine.push(frame)?,
        Request::Advance { budget } => engine.advance(budget),
        Request::End => engine.end()?,
        Request::Probe {
            sequence,
            state,
            cue,
        } => return engine.probe(sequence, state, cue),
        _ => {}
    }
    Ok(engine.snapshot())
}
#[cfg(test)]
mod tests;
