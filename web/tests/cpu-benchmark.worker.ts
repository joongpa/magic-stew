import type * as Ort from "onnxruntime-web";
const scope = self as unknown as {
  onmessage: (event: MessageEvent) => void;
  postMessage: (value: unknown) => void;
};
scope.onmessage = async ({ data }) => {
  let session: Ort.InferenceSession | undefined;
  let input: Ort.Tensor | undefined;
  try {
    const ort =
      data.runtime === "legacy"
        ? await import("onnxruntime-web/webgpu")
        : await import("onnxruntime-web/wasm");
    ort.env.wasm.wasmPaths = new URL("/runtime/", data.baseUrl).href;
    ort.env.wasm.numThreads = data.threads;
    const response = await fetch(
      new URL(`/models/cube-pose-${data.size}.onnx`, data.baseUrl),
    );
    if (!response.ok) throw new Error("Missing exported model");
    session = await ort.InferenceSession.create(await response.arrayBuffer(), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    const source = await fetch(
      new URL(`/models/cpu-bench-input-${data.size}.f32`, data.baseUrl),
    );
    if (!source.ok)
      throw new Error(
        "Run scripts/benchmark_web_cpu.py to generate the shared benchmark inputs.",
      );
    input = new ort.Tensor(
      "float32",
      new Float32Array(await source.arrayBuffer()),
      [1, 3, data.size, data.size],
    );
    const run = async () => {
      const start = performance.now();
      const output = await session!.run({ [session!.inputNames[0]]: input! });
      const ms = performance.now() - start;
      Object.values(output).forEach((t) => t.dispose());
      return ms;
    };
    for (let i = 0; i < 3; i++) await run();
    const timesMs = [];
    for (let i = 0; i < 9; i++) timesMs.push(await run());
    const ordered = [...timesMs].sort((a, b) => a - b);
    scope.postMessage({
      runtime: data.runtime,
      size: data.size,
      requestedThreads: data.threads,
      threads: ort.env.wasm.numThreads,
      isolated: self.crossOriginIsolated,
      medianMs: ordered[4],
      timesMs,
    });
  } catch (error) {
    scope.postMessage({
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    input?.dispose();
    await session?.release();
  }
};
