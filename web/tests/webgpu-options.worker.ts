import * as ort from "onnxruntime-web/webgpu";
const scope = self as unknown as {
  onmessage: (event: MessageEvent) => void;
  postMessage: (value: unknown) => void;
};
scope.onmessage = async ({ data }) => {
  let session: ort.InferenceSession | undefined;
  let inputBuffer: GPUBuffer | undefined,
    outputBuffer: GPUBuffer | undefined,
    readBuffer: GPUBuffer | undefined;
  let cpuInput: ort.Tensor | undefined;
  try {
    ort.env.wasm.wasmPaths = new URL("/runtime/", data.baseUrl).href;
    ort.env.wasm.numThreads = 1;
    const bytes = await (
      await fetch(new URL(`/models/cube-pose-${data.size}.onnx`, data.baseUrl))
    ).arrayBuffer();
    const gpuIo = data.mode !== "cpu-io";
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
      enableGraphCapture: data.mode === "capture",
      ...(gpuIo ? { preferredOutputLocation: "gpu-buffer" as const } : {}),
    });
    const device = ort.env.webgpu.device;
    const count = 3 * data.size * data.size;
    const values = new Float32Array(count);
    let seed = 17;
    for (let i = 0; i < count; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      values[i] = seed / 4294967296;
    }
    const dims = [1, 3, data.size, data.size];
    const outputDims = [
      1,
      16,
      [8, 16, 32].reduce((sum, stride) => sum + (data.size / stride) ** 2, 0),
    ];
    const outputBytes = outputDims.reduce((a, b) => a * b, 1) * 4;
    let input: ort.Tensor, fetches: Record<string, ort.Tensor> | undefined;
    if (gpuIo) {
      inputBuffer = device.createBuffer({
        size: values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      outputBuffer = device.createBuffer({
        size: outputBytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      readBuffer = device.createBuffer({
        size: outputBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      input = ort.Tensor.fromGpuBuffer(inputBuffer, {
        dataType: "float32",
        dims,
      });
      fetches = {
        [session.outputNames[0]]: ort.Tensor.fromGpuBuffer(outputBuffer, {
          dataType: "float32",
          dims: outputDims,
        }),
      };
    } else {
      cpuInput = new ort.Tensor("float32", values, dims);
      input = cpuInput;
    }
    let finalValues: Float32Array | undefined;
    const run = async () => {
      const start = performance.now();
      if (inputBuffer) device.queue.writeBuffer(inputBuffer, 0, values);
      const outputs = await session!.run(
        { [session!.inputNames[0]]: input },
        fetches,
      );
      if (gpuIo) {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(
          outputBuffer!,
          0,
          readBuffer!,
          0,
          outputBytes,
        );
        device.queue.submit([encoder.finish()]);
        await readBuffer!.mapAsync(GPUMapMode.READ);
        finalValues = new Float32Array(readBuffer!.getMappedRange()).slice();
        readBuffer!.unmap();
      } else {
        finalValues = (
          outputs[session!.outputNames[0]].data as Float32Array
        ).slice();
        Object.values(outputs).forEach((t) => t.dispose());
      }
      return performance.now() - start;
    };
    for (let i = 0; i < 4; i++) await run();
    const timesMs = [];
    for (let i = 0; i < 11; i++) timesMs.push(await run());
    const ordered = [...timesMs].sort((a, b) => a - b);
    scope.postMessage({
      size: data.size,
      mode: data.mode,
      medianMs: ordered[5],
      timesMs,
      shaderF16: device.features.has("shader-f16"),
      output: Array.from(finalValues!),
    });
  } catch (error) {
    scope.postMessage({
      size: data.size,
      mode: data.mode,
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    cpuInput?.dispose();
    await session?.release();
    inputBuffer?.destroy();
    outputBuffer?.destroy();
    readBuffer?.destroy();
  }
};
