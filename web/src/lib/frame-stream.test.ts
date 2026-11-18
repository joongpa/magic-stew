import assert from "node:assert/strict";
import test from "node:test";
import { FrameQueue, FrameSelector, FramePrefetcher } from "./frame-stream";

test("a trim boundary excludes earlier frames before establishing the 30fps sampling phase", () => {
  const selector = new FrameSelector(60, 30, 5.8);
  assert.deepEqual([5.784049,5.800716,5.817383,5.834049].map(t=>selector.accept(t)),[false,true,false,true]);
});

function selectedIndices(fps: number, count: number) {
  const selector = new FrameSelector(fps);
  return Array.from({ length: count }, (_, i) => i).filter((i) =>
    selector.accept(Math.round((i / fps) * 1e6) / 1e6),
  );
}

test("all frames are admitted at 24, 25, 29.97 and 30fps", () => {
  for (const fps of [24, 25, 29.97, 30])
    assert.equal(selectedIndices(fps, 300).length, 300);
});

test("60fps selects exactly frames 0, 2, 4… rather than whichever frame is newest after inference", () => {
  assert.deepEqual(
    selectedIndices(60, 120),
    Array.from({ length: 60 }, (_, i) => i * 2),
  );
});

test("120fps selects every fourth frame, and 50fps selects 30 frames per second", () => {
  assert.deepEqual(
    selectedIndices(120, 120),
    Array.from({ length: 30 }, (_, i) => i * 4),
  );
  assert.equal(selectedIndices(50, 50).length, 30);
});

test("source-time selection is unaffected by a slow asynchronous consumer", async () => {
  const selector = new FrameSelector(60);
  const seen: number[] = [];
  async function* frames() {
    for (let i = 0; i < 24; i++) if (selector.accept(i / 60)) yield i;
  }
  for await (const frame of frames()) {
    // Real scheduling delay: consumption is slower than 60fps. No newer frame replaces this one.
    await new Promise((resolve) => setTimeout(resolve, 40));
    seen.push(frame);
  }
  assert.deepEqual(
    seen,
    Array.from({ length: 12 }, (_, i) => i * 2),
  );
});

test("FIFO retains every admitted camera frame in arrival order while inference is busy", () => {
  const queue = new FrameQueue<{ index: number; close(): void }>();
  let closed = 0;
  for (let i = 0; i < 60; i++)
    queue.push({
      index: i,
      close() {
        closed++;
      },
    });
  for (let i = 0; i < 60; i++) assert.equal(queue.shift()?.index, i);
  assert.equal(queue.length, 0);
  assert.equal(closed, 0);
  queue.push({
    index: 60,
    close() {
      closed++;
    },
  });
  queue.clear();
  assert.equal(closed, 1);
});

// Lookahead owns at most one frame; slower inference must not replace it.
test("one-frame prefetch overlaps decoding and preserves every selected frame", async () => {
  let reads = 0;
  const closed: number[] = [];
  const source = new FramePrefetcher(async () => {
    const id = reads++;
    return id < 4 ? { id, close: () => closed.push(id) } : null;
  });
  const seen: number[] = [];
  for (let i = 0; i < 4; i++) {
    const frame = await source.next();
    assert.ok(frame);
    seen.push(frame.id);
    assert.equal(reads, i + 2);
    await Promise.resolve();
    frame.close();
  }
  assert.equal(await source.next(), null);
  assert.deepEqual(seen, [0, 1, 2, 3]);
  assert.deepEqual(closed, seen);
  source.close();
});
test("seeking disposes lookahead and does not return a stale frame", async () => {
  let resolveOld!: (frame: { id: number; close(): void }) => void;
  let reads = 0,
    closed = 0;
  const source = new FramePrefetcher(async () => {
    if (reads++ === 0)
      return new Promise<{ id: number; close(): void }>((resolve) => {
        resolveOld = resolve;
      });
    return { id: 100, close: () => closed++ };
  });
  const oldNext = source.next();
  source.reset();
  resolveOld({ id: 0, close: () => closed++ });
  assert.equal(await oldNext, null);
  assert.equal(closed, 1);
  const frame = await source.next();
  assert.equal(frame?.id, 100);
  await Promise.resolve();
  source.close(); // Dispose ready lookahead, while the delivered frame stays owned by its caller.
  assert.equal(closed, 2);
  frame?.close();
  assert.equal(closed, 3);
  assert.equal(await source.next(), null);
});
test("prefetch errors are delivered to the next consumer", async () => {
  let reads = 0;
  const failure = new Error("decode failed");
  const source = new FramePrefetcher(async () => {
    if (reads++) throw failure;
    return { close() {} };
  });
  (await source.next())?.close();
  await Promise.resolve();
  await assert.rejects(source.next(), failure);
  source.close();
});
