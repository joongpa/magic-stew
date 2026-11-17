import * as current from "../src/lib/preprocess";
import * as reference from "./resize-reference";
const scope = self as unknown as {
  onmessage: () => void;
  postMessage: (value: unknown) => void;
};
const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
scope.onmessage = () => {
  try {
    const results = [];
    for (const [kind, width, height, size] of [
      ["pose", 608, 608, 160],
      ["pose", 591, 591, 160],
      ["pose", 608, 608, 256],
      ["pose", 721, 719, 288],
      ["linear", 1280, 720, 640],
    ] as const) {
      const input = new Uint8ClampedArray(width * height * 4);
      let seed = 17;
      for (let i = 0; i < input.length; i++) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        input[i] = seed >>> 24;
      }
      const run = (impl: typeof current) =>
        kind === "pose"
          ? impl.posePixels(input, width, height, 256, size)
          : impl.resizeLinear(input, width, height, size, 360);
      const expected = run(reference),
        actual = run(current);
      let maxDifference = 0;
      for (let i = 0; i < actual.length; i++)
        maxDifference = Math.max(
          maxDifference,
          Math.abs(actual[i] - expected[i]),
        );
      for (let i = 0; i < 8; i++) {
        run(reference);
        run(current);
      }
      const oldTimes: number[] = [],
        newTimes: number[] = [];
      const time = (impl: typeof current, values: number[]) => {
        const start = performance.now();
        run(impl);
        values.push(performance.now() - start);
      };
      for (let i = 0; i < 21; i++) {
        if (i % 2) {
          time(current, newTimes);
          time(reference, oldTimes);
        } else {
          time(reference, oldTimes);
          time(current, newTimes);
        }
      }
      const beforeMs = median(oldTimes),
        afterMs = median(newTimes);
      const result = {
        kind,
        width,
        height,
        size,
        beforeMs,
        afterMs,
        speedup: beforeMs / afterMs,
        maxDifference,
        oldTimes,
        newTimes,
      };
      results.push(result);
      scope.postMessage({ type: "result", result });
    }
    scope.postMessage({ type: "done", results });
  } catch (error) {
    scope.postMessage({ type: "error", message: String(error) });
  }
};
