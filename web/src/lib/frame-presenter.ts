export type OverlayPainter = (
  context: OffscreenCanvasRenderingContext2D,
  width: number,
  height: number,
) => void;

/** Keeps the visible canvas on a completed frame until its successor is ready. */
export class FramePresenter {
  private pending?: OffscreenCanvas;
  private displayed?: OffscreenCanvas;
  private composite?: OffscreenCanvas;

  get snapshot() {
    return this.displayed;
  }

  capture(source: HTMLImageElement | ImageBitmap | OffscreenCanvas) {
    const width =
      source instanceof HTMLImageElement ? source.naturalWidth : source.width;
    const height =
      source instanceof HTMLImageElement ? source.naturalHeight : source.height;
    if (
      !this.pending ||
      this.pending.width !== width ||
      this.pending.height !== height
    )
      this.pending = new OffscreenCanvas(width, height);
    const context = this.pending.getContext("2d")!;
    context.clearRect(0, 0, width, height);
    context.drawImage(source, 0, 0);
  }

  present(target: HTMLCanvasElement, overlay?: OverlayPainter) {
    if (!this.pending) return;
    // The old raw frame becomes the next capture buffer. Overlay toggles always
    // redraw the displayed raw frame, even while another frame is in flight.
    [this.displayed, this.pending] = [this.pending, this.displayed];
    this.redraw(target, overlay);
  }

  redraw(target: HTMLCanvasElement, overlay?: OverlayPainter) {
    if (!this.displayed) return;
    const source = this.displayed;
    const scale = Math.min(1, 1920 / Math.max(source.width, source.height));
    const width = Math.round(source.width * scale),
      height = Math.round(source.height * scale);
    let image = source;
    if (overlay) {
      if (
        !this.composite ||
        this.composite.width !== width ||
        this.composite.height !== height
      )
        this.composite = new OffscreenCanvas(width, height);
      const context = this.composite.getContext("2d")!;
      context.clearRect(0, 0, width, height);
      context.drawImage(source, 0, 0, width, height);
      overlay(context, width, height);
      image = this.composite;
    }
    // Only this final blit touches the visible frame. Resizing happens after
    // composition too, so a new source size cannot clear the old view early.
    if (target.width !== width) target.width = width;
    if (target.height !== height) target.height = height;
    const visible = target.getContext("2d")!;
    visible.globalCompositeOperation = "copy";
    visible.drawImage(image, 0, 0, width, height);
    visible.globalCompositeOperation = "source-over";
  }

  clear(target?: HTMLCanvasElement) {
    this.pending = undefined;
    this.displayed = undefined;
    this.composite = undefined;
    target?.getContext("2d")?.clearRect(0, 0, target.width, target.height);
  }
}
