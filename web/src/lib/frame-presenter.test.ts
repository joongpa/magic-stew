import assert from "node:assert/strict";
import test from "node:test";
import { FramePresenter } from "./frame-presenter";

test('cropped presentation copies the selected original pixels without stretching', t => {
  const draws: unknown[][] = [];
  class Canvas {
    constructor(public width: number, public height: number) {}
    getContext() { return {clearRect() {}, drawImage(...args: unknown[]) {draws.push(args);}}; }
  }
  const globals = globalThis as unknown as Record<string, unknown>;
  const oldCanvas = globals.OffscreenCanvas, oldImage = globals.HTMLImageElement;
  globals.OffscreenCanvas = Canvas; globals.HTMLImageElement = class {};
  t.after(() => {globals.OffscreenCanvas = oldCanvas; globals.HTMLImageElement = oldImage;});
  const source = new Canvas(3840, 2160), presenter = new FramePresenter();
  const frame = presenter.captureFrame(source as unknown as OffscreenCanvas, {x: 960, y: 270, width: 1920, height: 1620});
  assert.equal(frame.width, 1920); assert.equal(frame.height, 1620);
  assert.deepEqual(draws[0], [source, 960, 270, 1920, 1620, 0, 0, 1920, 1620]);
  presenter.releaseFrame(frame);
  const full = presenter.captureFrame(source as unknown as OffscreenCanvas);
  assert.equal(full.width, 3840); assert.equal(full.height, 2160);
  assert.deepEqual(draws[1], [source, 0, 0]);
});

test("buffered raw frames stay separate and each overlay is composed with its own frame", (t) => {
  class Canvas {
    pixel = 0;
    context = {
      canvas: this,
      globalCompositeOperation: "source-over",
      clearRect: () => { this.pixel = 0; },
      drawImage: (source: Canvas) => { this.pixel = source.pixel; },
    };
    constructor(public width: number, public height: number) {}
    getContext() { return this.context; }
  }
  const globals = globalThis as unknown as Record<string, unknown>;
  const oldCanvas = globals.OffscreenCanvas, oldImage = globals.HTMLImageElement;
  globals.OffscreenCanvas = Canvas;
  globals.HTMLImageElement = class {};
  t.after(() => {
    globals.OffscreenCanvas = oldCanvas;
    globals.HTMLImageElement = oldImage;
  });
  const presenter = new FramePresenter(), visible = new Canvas(32, 32);
  const capture = (pixel: number) => {
    const input = new Canvas(32, 32); input.pixel = pixel;
    return presenter.captureFrame(input as unknown as OffscreenCanvas);
  };
  const frameA = capture(100), frameB = capture(200);
  assert.notEqual(frameA, frameB);
  const paint = (amount: number) => (context: OffscreenCanvasRenderingContext2D) => {
    (context.canvas as unknown as Canvas).pixel += amount;
  };
  presenter.presentFrame(visible as unknown as HTMLCanvasElement, frameA, paint(10));
  assert.equal(visible.pixel, 110);
  assert.equal((presenter.snapshot as unknown as Canvas).pixel, 100);
  presenter.presentFrame(visible as unknown as HTMLCanvasElement, frameB, paint(20));
  assert.equal(visible.pixel, 220);
  const frameC = capture(300);
  assert.equal((presenter.snapshot as unknown as Canvas).pixel, 200);
  presenter.redraw(visible as unknown as HTMLCanvasElement, paint(20));
  assert.equal(visible.pixel, 220);
  presenter.releaseFrame(frameC);
});
