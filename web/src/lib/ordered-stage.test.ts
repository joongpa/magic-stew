import assert from "node:assert/strict";
import test from "node:test";
import { OrderedStage } from "./ordered-stage";

function gate() {
  let open!: () => void;
  const promise = new Promise<void>((resolve) => { open = resolve; });
  return { promise, open };
}

test("downstream fitting allows upstream admissions, applies backpressure, and completes every frame in order", async () => {
  const gates = [gate(), gate(), gate()];
  const started: number[] = [], finished: number[] = [], released: number[] = [];
  const stage = new OrderedStage<number>(3, async (frame) => {
    started.push(frame);
    await gates[frame].promise;
    finished.push(frame);
  }, (frame) => { released.push(frame); });
  stage.enqueue(0);
  // Producer can admit N+1 while fitting N is still blocked.
  assert.equal(stage.canAdmit, true);
  stage.enqueue(1); stage.enqueue(2);
  assert.deepEqual(started, [0]);
  assert.equal(stage.size, 3); assert.equal(stage.canAdmit, false);
  assert.throws(() => stage.enqueue(3), /capacity/);
  gates[0].open(); await Promise.resolve(); await Promise.resolve();
  assert.equal(stage.canAdmit, true);
  assert.deepEqual(started, [0, 1]);
  gates[1].open(); gates[2].open(); await stage.whenIdle();
  assert.deepEqual(finished, [0, 1, 2]); assert.deepEqual(released, [0, 1, 2]);
});

test("source reset releases queued buffers while an active fitting job finishes", async () => {
  const first = gate(), seen: number[] = [], released: number[] = [];
  const stage = new OrderedStage<number>(3, async (frame) => {
    seen.push(frame); if (frame === 0) await first.promise;
  }, (frame) => { released.push(frame); });
  stage.enqueue(0); stage.enqueue(1); stage.enqueue(2);
  stage.clear(); assert.deepEqual(released, [1, 2]);
  stage.enqueue(3); first.open(); await stage.whenIdle();
  assert.deepEqual(seen, [0, 3]); assert.deepEqual(released, [1, 2, 0, 3]);
});

test("a failed job releases its buffer and cannot deadlock the next frame", async () => {
  const seen: number[] = [], errors: unknown[] = [], released: number[] = [];
  const stage = new OrderedStage<number>(3, async (frame) => {
    if (!frame) throw new Error("fit failed"); seen.push(frame);
  }, (frame) => { released.push(frame); }, () => {}, (error) => { errors.push(error); });
  stage.enqueue(0); stage.enqueue(1); await stage.whenIdle();
  assert.equal(errors.length, 1); assert.deepEqual(seen, [1]); assert.deepEqual(released, [0, 1]);
});
