import { GpuPreprocessor } from "../src/lib/gpu-preprocess";
import { posePixels, resizeLinear, paddedPixels } from "../src/lib/preprocess";
import { rgbaToChw, letterbox, roiTransform } from "../src/lib/detector";
const scope = self as unknown as {
  onmessage: () => void;
  postMessage: (v: unknown) => void;
};
async function read(device: GPUDevice, buffer: GPUBuffer, count: number) {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return values;
}
scope.onmessage = async () => {
  let gpu: GpuPreprocessor | undefined, device: GPUDevice | undefined;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter");
    device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    gpu = await GpuPreprocessor.create(device);
    const results = [];
    for (const [kind, width, height, crop, model, alpha] of [
      ["pose", 640, 640, 608, 160, false],
      ["pose", 640, 640, 591, 256, false],
      ["pose", 760, 750, 721, 288, false],
      ["pose", 540, 540, 512, 160, false],
      ["pose", 280, 280, 256, 256, false],
      ["pose", 170, 170, 144, 160, false],
      ["pose", 40, 40, 17, 160, false],
      ["pose", 640, 640, 591, 160, true],
      ["pose", 280, 280, 256, 256, true],
      ["bbox", 1280, 720, 0, 640, false],
      ["bbox", 720, 1280, 0, 640, false],
      ["bbox", 17, 13, 0, 32, false],
    ] as const) {
      const canvas = new OffscreenCanvas(width, height),
        ctx = canvas.getContext("2d")!;
      const pixels = ctx.createImageData(width, height);
      let seed = 17;
      for (let i = 0; i < pixels.data.length; i++) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        pixels.data[i] = i % 4 === 3 && !alpha ? 255 : seed >>> 24;
      }
      ctx.putImageData(pixels, 0, 0);
      const bitmap = await createImageBitmap(canvas);
      const t =
        kind === "pose"
          ? roiTransform(width, height, model, [7, 9, 7 + crop, 9 + crop])
          : letterbox(width, height, model, false, 32);
      const cropped = new OffscreenCanvas(t.cropWidth, t.cropHeight),
        c = cropped.getContext("2d", { willReadFrequently: true })!;
      const cpu = () => {
        c.drawImage(
          bitmap,
          t.cropX,
          t.cropY,
          t.cropWidth,
          t.cropHeight,
          0,
          0,
          t.cropWidth,
          t.cropHeight,
        );
        const rgba = c.getImageData(0, 0, t.cropWidth, t.cropHeight).data;
        return rgbaToChw(
          kind === "pose"
            ? posePixels(rgba, t.cropWidth, t.cropHeight, 256, model)
            : paddedPixels(
                resizeLinear(
                  rgba,
                  t.cropWidth,
                  t.cropHeight,
                  t.resizedWidth,
                  t.resizedHeight,
                ),
                t.resizedWidth,
                t.resizedHeight,
                t.inputWidth,
                t.inputHeight,
                t.padX,
                t.padY,
              ),
        );
      };
      const expected = cpu(),
        prepared = gpu.prepare(bitmap, t, kind === "pose" ? 256 : undefined);
      const actual = await read(device, prepared.buffer, expected.length);
      let maxPixelDifference = 0,
        maxTensorDifference = 0,
        different = 0,
        firstDifference;
      for (let i = 0; i < expected.length; i++)
        if (expected[i] !== actual[i]) {
          different++;
          maxTensorDifference = Math.max(
            maxTensorDifference,
            Math.abs(expected[i] - actual[i]),
          );
          maxPixelDifference = Math.max(
            maxPixelDifference,
            Math.abs(
              Math.round(expected[i] * 255) - Math.round(actual[i] * 255),
            ),
          );
          firstDifference ??= { i, expected: expected[i], actual: actual[i] };
        }
      const cpuTimes: number[] = [],
        gpuTimes: number[] = [];
      for (let i = 0; i < 12; i++) {
        let mark = performance.now();
        cpu();
        const a = performance.now() - mark;
        mark = performance.now();
        gpu.prepare(bitmap, t, kind === "pose" ? 256 : undefined);
        await device.queue.onSubmittedWorkDone();
        const b = performance.now() - mark;
        if (i >= 3) {
          cpuTimes.push(a);
          gpuTimes.push(b);
        }
      }
      const median = (a: number[]) =>
        a.sort((a, b) => a - b)[Math.floor(a.length / 2)];
      const result = {
        kind,
        width,
        height,
        crop,
        model,
        alpha,
        different,
        maxPixelDifference,
        maxTensorDifference,
        firstDifference,
        cpuMs: median(cpuTimes),
        gpuMs: median(gpuTimes),
      };
      results.push(result);
      scope.postMessage({ type: "result", result });
      bitmap.close();
    }
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
    scope.postMessage({ type: "done", results });
  } catch (error) {
    scope.postMessage({ type: "error", message: String(error) });
  } finally {
    gpu?.dispose();
    device?.destroy();
  }
};
