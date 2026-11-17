import assert from "node:assert/strict";
import test from "node:test";
import { FramePresenter } from "./frame-presenter";

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
