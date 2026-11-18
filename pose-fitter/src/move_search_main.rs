use cube_pose_fitter::move_search::{Config, Engine, Frame};
use serde::Deserialize;
use std::{fs, time::Instant};
#[derive(Deserialize)]
struct Input {
    config: Config,
    frames: Vec<Frame>,
    #[serde(default)]
    probes: Vec<Probe>,
}
#[derive(Deserialize)]
struct Probe {
    sequence: usize,
    state: Vec<u8>,
    cue: Option<usize>,
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let data: Input = serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();
    let mut engine = Engine::new(data.config).unwrap();
    let start = Instant::now();
    for frame in data.frames {
        engine.push(frame).unwrap();
    }
    engine.end().unwrap();
    let prepare_seconds = start.elapsed().as_secs_f64();
    let probes: Vec<_> = data
        .probes
        .into_iter()
        .map(|p| engine.probe(p.sequence, p.state, p.cue).unwrap())
        .collect();
    let start = Instant::now();
    loop {
        engine.advance(10000);
        if engine.snapshot()["status"] != "running" {
            break;
        }
    }
    let out = serde_json::json!({"result":engine.snapshot(),"probes":probes,"prepare_seconds":prepare_seconds,"search_seconds":start.elapsed().as_secs_f64()});
    fs::write(&args[2], serde_json::to_vec(&out).unwrap()).unwrap();
    println!("{}", engine.snapshot()["status"]);
}
