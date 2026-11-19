/** Replay a downloaded app recording with the batch CPU engine, without a UI. */
import { readFileSync, writeFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import { createHash } from "node:crypto";
import { pipelineFrame } from "../../web/src/lib/pose-fitter";
import { loadCpuFitter } from "../runtime/wasm-host.mjs";
// Generated Emscripten module.
// @ts-ignore
import createOpenCv from "../target/opencv-pnp.mjs";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) throw new Error("Usage: web/node_modules/.bin/tsx pose-fitter/tools/check-browser-recording.mts recording.json report.json");
const recording = JSON.parse(readFileSync(inputPath, "utf8"));
const bytes = readFileSync(new URL("../target/parity-wasm/wasm32-unknown-unknown/release/cube_pose_fitter.wasm", import.meta.url));
const geometry = JSON.parse(readFileSync(new URL("../tests/fixtures/cpu-golden.json", import.meta.url), "utf8")).geometry;
const engine = await loadCpuFitter(bytes, Buffer.from(JSON.stringify({ geometry, workloads: [], probes: [] })), await createOpenCv());
const groups = new Map<string, any[]>();
for (const row of recording.frames) {
  if (!row.result.fittedPose) continue;
  const key = `${row.sourceId}/${row.segment}/${row.result.fittedPose.fittingConfig?.segment ?? 1}`;
  const rows = groups.get(key) ?? []; rows.push(row); groups.set(key, rows);
}
const issues: any[] = [], segments: any[] = [];
let checked = 0;
const expectedVersion = `cpu-pipeline.${createHash("sha256").update(bytes).digest("hex").slice(0, 16)}.wasm`;
for (const [segment, rows] of groups) {
  const config = rows[0].result.fittedPose.fittingConfig;
  if (!config) throw new Error("Recording lacks fittingConfig. Re-run it using the complete pipeline app.");
  const frames = rows.map(row => pipelineFrame(row.result, row.result.fittedPose.sourceFrame));
  const batch = engine.call("pipeline_json", { frames, fps: config.fps,
    turning_start_frame: config.turningStartFrame, expected_center_colors: config.centers });
  if (batch.Err) throw new Error(batch.Err);
  for (let i = 0; i < rows.length; i++) {
    const actual = rows[i].result.fittedPose, reference = batch.Ok.frames[i];
    if (actual.stage !== "cpu_pipeline" || actual.pipelineVersion !== expectedVersion ||
        !isDeepStrictEqual(actual.bank, reference.bank) || actual.depth_recovery !== reference.depth_recovery ||
        !isDeepStrictEqual(actual.startup_reference_core, reference.startup_reference_core) ||
        actual.startup_motion_allowance !== reference.startup_motion_allowance ||
        (actual.pose_flag_reason != null && actual.continuity_core != null))
      issues.push({ segment, frame: frames[i].frame });
  }
  checked += rows.length;
  segments.push({ segment, frames: rows.length, first: frames[0].frame, last: frames.at(-1)!.frame });
}
if (!checked) throw new Error("No fitted frames in the recording");
const report = { recording: inputPath, checked, segments, expectedVersion, differingFrames: issues };
writeFileSync(outputPath, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report));
if (issues.length) process.exitCode = 1;
