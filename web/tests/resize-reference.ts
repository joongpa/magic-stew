/** Frozen pre-optimization resize kernels, used only for parity and benchmarks. */
const f32 = Math.fround;
export function roundEven(value: number): number {
  const floor = Math.floor(value);
  return value - floor === 0.5
    ? floor % 2 === 0
      ? floor
      : floor + 1
    : Math.round(value);
}

interface Tap {
  index: number;
  weight: number;
}
function areaAxis(source: number, target: number): Tap[][] {
  const scale = source / target;
  return Array.from({ length: target }, (_, i) => {
    const start = i * scale,
      end = (i + 1) * scale;
    const taps: Tap[] = [];
    for (let p = Math.floor(start); p < Math.ceil(end); p++) {
      const weight = (Math.min(end, p + 1) - Math.max(start, p)) / scale;
      if (weight > 1e-9)
        taps.push({
          index: Math.min(source - 1, Math.max(0, p)),
          weight: f32(weight),
        });
    }
    return taps;
  });
}

export function resizeArea(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number,
): Uint8ClampedArray {
  if (width === targetWidth && height === targetHeight) return rgba;
  const xs = areaAxis(width, targetWidth),
    ys = areaAxis(height, targetHeight);
  const horizontal = new Float32Array(height * targetWidth * 3);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < targetWidth; x++)
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (const tap of xs[x])
          sum = f32(
            sum + f32(rgba[(y * width + tap.index) * 4 + c] * tap.weight),
          );
        horizontal[(y * targetWidth + x) * 3 + c] = sum;
      }
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 4);
  const fast2 = width === targetWidth * 2 && height === targetHeight * 2;
  for (let y = 0; y < targetHeight; y++)
    for (let x = 0; x < targetWidth; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (const tap of ys[y])
          sum = f32(
            sum +
              f32(
                horizontal[(tap.index * targetWidth + x) * 3 + c] * tap.weight,
              ),
          );
        output[(y * targetWidth + x) * 4 + c] = fast2
          ? Math.floor(sum + 0.5)
          : roundEven(sum);
      }
      output[(y * targetWidth + x) * 4 + 3] = 255;
    }
  return output;
}

function linearAxis(source: number, target: number) {
  return Array.from({ length: target }, (_, i) => {
    let at = f32(((i + 0.5) * source) / target - 0.5),
      index = Math.floor(at),
      weight = f32(at - index);
    if (index < 0) {
      index = 0;
      weight = 0;
    }
    if (index >= source - 1) {
      index = source - 1;
      weight = 0;
    }
    return {
      a: index,
      b: Math.min(source - 1, index + 1),
      wa: roundEven(f32((1 - weight) * 2048)),
      wb: roundEven(f32(weight * 2048)),
    };
  });
}
export function resizeLinear(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number,
): Uint8ClampedArray {
  if (width === targetWidth && height === targetHeight) return rgba;
  const xs = linearAxis(width, targetWidth),
    ys = linearAxis(height, targetHeight);
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 4);
  for (let y = 0; y < targetHeight; y++)
    for (let x = 0; x < targetWidth; x++) {
      const xx = xs[x],
        yy = ys[y];
      for (let c = 0; c < 3; c++) {
        const top =
          rgba[(yy.a * width + xx.a) * 4 + c] * xx.wa +
          rgba[(yy.a * width + xx.b) * 4 + c] * xx.wb;
        const bottom =
          rgba[(yy.b * width + xx.a) * 4 + c] * xx.wa +
          rgba[(yy.b * width + xx.b) * 4 + c] * xx.wb;
        output[(y * targetWidth + x) * 4 + c] = Math.floor(
          (top * yy.wa + bottom * yy.wb + 2097152) / 4194304,
        );
      }
      output[(y * targetWidth + x) * 4 + 3] = 255;
    }
  return output;
}

/** Native pose path: ROI --INTER_AREA--> viewSize --INTER_LINEAR--> model size. */
export function posePixels(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  viewSize: number,
  size: number,
) {
  return resizeLinear(
    resizeArea(rgba, width, height, viewSize, viewSize),
    viewSize,
    viewSize,
    size,
    size,
  );
}
export function paddedPixels(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  inputWidth: number,
  inputHeight: number,
  padX: number,
  padY: number,
) {
  if (width === inputWidth && height === inputHeight) return rgba;
  const output = new Uint8ClampedArray(inputWidth * inputHeight * 4);
  for (let i = 0; i < output.length; i += 4) {
    output[i] = output[i + 1] = output[i + 2] = 114;
    output[i + 3] = 255;
  }
  for (let y = 0; y < height; y++)
    output.set(
      rgba.subarray(y * width * 4, (y + 1) * width * 4),
      ((y + padY) * inputWidth + padX) * 4,
    );
  return output;
}
