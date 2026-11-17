// Replay identical saved detections through the actual browser ABI, without
// feeding its tracker the reference cameras. Keep the fixed-input replay as a
// separate control; it cannot establish autonomous tracking parity.
import { readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { loadFitter } from './wasm-host.mjs';

const [fixturePath, outputPath, strideText = '2', coldFramesText = ''] = process.argv.slice(2);
const fixtureBytes = readFileSync(fixturePath);
const fixture = JSON.parse(fixtureBytes);
const binary = readFileSync(new URL('../target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm', import.meta.url));
const engine = await loadFitter(binary, fixtureBytes);
const stride = Number(strideText);
if (!Number.isInteger(stride) || stride < 1) throw new Error('Stride must be a positive integer');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const coldFrames = new Set(coldFramesText.split(',').filter(Boolean).map(Number));
const result = { scope: 'Same saved detections: browser rollout versus fixed-input CPU reference. No detector inference.', stride,
  wasm_sha256: hash(binary), fixture_sha256: hash(fixtureBytes), rows: [] };
let previous = null, ordinal = 0;
for (let index = 0; index < fixture.workloads.length; index++) {
  const item = fixture.workloads[index];
  if (item.stage !== 'initial_geometry' || ordinal++ % stride !== 0) continue;
  const started = performance.now();
  const browser = engine.fitBrowser({ width: item.image_size[0], height: item.image_size[1], observations: item.observations, previous });
  previous = browser.pose;
  engine.run(index);
  const fixed = engine.result();
  result.rows.push({ frame: item.frame, browser, fixed: fixed.bank, elapsedMs: performance.now() - started });
  if (result.rows.length % 30 === 0) console.log('rollout', result.rows.length, 'frame', item.frame);
}
// Use the immutable initial winners on both sides, matching the CPU second
// pass. A temporal result must not become another frame's neighbor camera.
const initial = new Map(fixture.workloads.filter(w => w.stage === 'initial_geometry').map(w => [w.frame, w]));
for (let index = 0; index < result.rows.length; index++) {
  const row = result.rows[index], item = initial.get(row.frame);
  const neighbors = [result.rows[index - 1]?.browser.pose, result.rows[index + 1]?.browser.pose].filter(Boolean);
  const input = { width: item.image_size[0], height: item.image_size[1], observations: item.observations };
  row.revised = engine.refineBrowser({ ...input, bank: row.browser.bank, neighbors });
  if (coldFrames.has(row.frame)) row.cold = engine.fitBrowser({ ...input, previous: null });
  if (index % 40 === 0) console.log('neighbor pass', index + 1, 'frame', row.frame);
}
writeFileSync(outputPath, JSON.stringify(result));
console.log('saved', result.rows.length, outputPath);
