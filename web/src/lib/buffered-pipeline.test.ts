import assert from "node:assert/strict";
import test from "node:test";
import { BufferedPosePipeline, type PipelineFrame } from "../../../pose-fitter/runtime/buffered-pipeline.mjs";

const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const frame = (index: number): PipelineFrame => ({ frame: index, width: 800, height: 600, pixel_scale: 1,
  observations: { points: [[0, 0], [1, 0], [0, 1]], classes: [0, 1, 2], stickers: [[1, 1]], uncertainties: [1] },
  colors: ["white"], landmark_presence: [1, 1, 1] });
const camera = (index: number) => ({ rvec: [index, 0, 0], tvec: [0, 0, 500] });
const fit = (index: number) => ({ params: [...camera(index).rvec, ...camera(index).tvec], axis: 0,
  angles: [0, 0, 0], score: 0, flat_fallback: true, nearest: [], valid: [] });
function executor(log: Record<string, any>[]) {
  return async (job: Record<string, any>) => {
    log.push(job);
    switch (job.type) {
      case "prepare": return { candidates: [camera(job.frame.frame)] };
      case "select_startup": return { source: job.candidates[0] ?? null,
        core: job.candidates.length ? [job.frame.frame, 0] : null, anchor: job.candidates.length > 0 };
      case "initial": return { bank: [fit(job.frame.frame)] };
      case "neighbors": return { bank: [fit(job.frame.frame + 100)] };
      case "orient": return { bank: job.bank, reference: { frame: job.frame.frame, rvec: [0, 0, 0] } };
      case "target": return { target: 500 };
      default: return { bank: job.bank, wireframe: [], pose: camera(job.frame.frame), depth_recovery: false,
        continuity_core: [job.frame.frame, 0] };
    }
  };
}
const config = { fps: 10, turningStartFrame: null, centers: [] };

test("buffered fitting admits future frames, preserves initial priors and flushes every tail frame", async () => {
  const log: Record<string, any>[] = [], stream = new BufferedPosePipeline(executor(log), config);
  let firstFinished = false;
  const first = stream.push(frame(0)).then(value => { firstFinished = true; return value; });
  const second = stream.push(frame(1));
  await tick();
  assert.equal(firstFinished, false, "recovery must wait for 180ms of future source context");
  const third = stream.push(frame(2));
  await tick();
  assert.equal(firstFinished, true);
  await stream.flush();
  const results = await Promise.all([first, second, third]);
  assert.deepEqual(results.map(r => r.pose?.rvec[0]), [0, 1, 2]);
  assert.deepEqual(log.filter(j => j.type === "initial").map(j => j.prior?.rvec[0] ?? null), [null, 0, 1]);
  assert.deepEqual(log.filter(j => j.type === "neighbors").map(j => j.neighbors.map((p: any) => p.rvec[0])), [[1], [0, 2], [1]]);
  assert.deepEqual(log.filter(j => j.type === "orient").map(j => j.reference?.frame ?? null), [null, 0, 1]);
  assert.deepEqual(log.filter(j => j.type === "target").map(j => j.frames.map((f: any) => f.frame)), [[0, 1], [0, 1, 2], [1, 2]]);
  assert.throws(() => stream.push(frame(3)), /closed/);
});

test("startup completion order cannot select a later seed or reorder orientation", async () => {
  const log: Record<string, any>[] = [], execute = executor(log);
  let release!: () => void;
  const delayed = new Promise<void>(resolve => { release = resolve; });
  const stream = new BufferedPosePipeline(async job => {
    if (job.type === "prepare" && (job.frame as PipelineFrame).frame === 1) await delayed;
    const result = await execute(job);
    if (job.type === "prepare" && (job.frame as PipelineFrame).frame === 0) return { candidates: [] };
    return result;
  }, config);
  const pending = [0, 1, 2].map(index => stream.push(frame(index)));
  await tick();
  assert.equal(log.some(j => j.type === "initial"), false);
  release(); await stream.flush(); await Promise.all(pending);
  assert.equal(log.find(j => j.type === "initial")!.source.rvec[0], 1);
  assert.deepEqual(log.filter(j => j.type === "orient").map(j => j.frame.frame), [0, 1, 2]);
});

test("startup selection follows source order and waits for finalized reference cores", async () => {
  const log: Record<string, any>[] = [], execute = executor(log);
  let release!: () => void;
  const delayed = new Promise<void>(resolve => { release = resolve; });
  const stream = new BufferedPosePipeline(async job => {
    const index = (job.frame as PipelineFrame | undefined)?.frame;
    if (job.type === "prepare" && index === 0) await delayed;
    const result = await execute(job);
    if (job.type === "prepare" && index === 1) return { candidates: [] };
    return result;
  }, config);
  const pending = [0, 1, 2].map(index => stream.push(frame(index)));
  await tick();
  assert.equal(log.some(j => j.type === "select_startup"), false);
  release(); await stream.flush(); await Promise.all(pending);
  const selections = log.filter(j => j.type === "select_startup");
  assert.deepEqual(selections.map(j => j.frame.frame), [0, 1, 2]);
  assert.deepEqual(selections.map(j => j.references), [[], [], [{frame:0,core:[0,0]}]]);
  const targets = log.filter(j => j.type === "target");
  assert.ok(targets.every(j => j.anchors.every((i: number) => j.frames[i].frame !== 1)));
});

test("flagged final poses cannot replace the reference, regardless of worker completion order", async () => {
  const run = async (delayed: boolean) => {
    const log: Record<string, any>[] = [], execute = executor(log);
    const stream = new BufferedPosePipeline(async job => {
      const index = (job.frame as PipelineFrame | undefined)?.frame;
      if (delayed && job.type === "finalize" && index === 0) await tick();
      const result = await execute(job);
      if (job.type === "finalize" && index !== 1 && index !== 2) return { ...result, continuity_core: null };
      return result;
    }, { ...config, fps: 30 });
    const pending = Array.from({length: 10}, (_,i)=>stream.push(frame(i)));
    await stream.flush(); await Promise.all(pending);
    return log.filter(j=>j.type === "select_startup").map(j=>j.references);
  };
  const one = [{frame:1,core:[1,0]}], two = [...one,{frame:2,core:[2,0]}];
  const expected = [[],[],[],[],[],[],[],one,two,two];
  assert.deepEqual(await run(false), expected);
  assert.deepEqual(await run(true), expected);
});

test("cancelling rejects outstanding frames and a pending flush", async () => {
  const stream = new BufferedPosePipeline(() => new Promise(() => {}), config);
  const pending = stream.push(frame(0));
  const flushed = stream.flush();
  stream.cancel(new Error("Source changed"));
  await assert.rejects(pending, /Source changed/);
  await assert.rejects(flushed, /Source changed/);
});
test('completed live fits retain only continuity cores internally while callers receive full banks',async()=>{
 const stream=new BufferedPosePipeline(executor([]),config);
 const pending=[0,1,2].map(i=>stream.push(frame(i)));await stream.flush();
 const results=await Promise.all(pending);
 assert.ok(results.every(result=>result.bank.length===1));
 for(const entry of (stream as any).entries){
  assert.deepEqual(Object.keys(await entry.result),['continuity_core']);assert.equal(entry.frame,null);
 }
});
