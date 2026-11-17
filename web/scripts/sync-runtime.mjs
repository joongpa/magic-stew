import { mkdir, copyFile, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";

// Serve the WASM bootstrap from the same origin, with exactly the JS package's version.
const require = createRequire(import.meta.url);
const dist = dirname(require.resolve("onnxruntime-web"));
const target = new URL("../public/runtime/", import.meta.url);
await mkdir(target, { recursive: true });
for (const file of [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
]) {
  await copyFile(join(dist, file), new URL(file, target));
}

// Content-addressed CPU pipeline assets prevent the old preview WASM from
// being reused after a fitter update. All workers load this same manifest.
try {
  const sources = {
    rust: new URL("../../pose-fitter/target/parity-wasm/wasm32-unknown-unknown/release/cube_pose_fitter.wasm", import.meta.url),
    opencvModule: new URL("../../pose-fitter/target/opencv-pnp.mjs", import.meta.url),
    opencvWasm: new URL("../../pose-fitter/target/opencv-pnp.wasm", import.meta.url),
  };
  const manifest = { version: 1, engine: "cpu_pipeline_v1" };
  const output = new URL("../public/pose/", import.meta.url);
  await mkdir(output, { recursive: true });
  for (const [key, source] of Object.entries(sources)) {
    const bytes = await readFile(source);
    const hash = createHash("sha256").update(bytes).digest("hex").slice(0, 16);
    const filename = key === "rust" ? `cpu-pipeline.${hash}.wasm`
      : key === "opencvModule" ? `opencv-pnp.${hash}.mjs` : `opencv-pnp.${hash}.wasm`;
    await writeFile(new URL(filename, output), bytes);
    manifest[key] = filename;
  }
  const golden = JSON.parse(await readFile(new URL("../../pose-fitter/tests/fixtures/cpu-golden.json", import.meta.url), "utf8"));
  const geometry = JSON.stringify(golden.geometry);
  const hash = createHash("sha256").update(geometry).digest("hex").slice(0, 16);
  manifest.geometry = `geometry.${hash}.json`;
  await writeFile(new URL(manifest.geometry, output), geometry);
  const licenses = new URL("licenses/", output);
  await mkdir(licenses, { recursive: true });
  for (const name of ["LICENSE", "README.md", "ippe.cpp", "sqpnp.cpp", "p3p.cpp", "ap3p.cpp"]) {
    await copyFile(new URL(`../../pose-fitter/vendor/opencv/${name}`, import.meta.url), new URL(name, licenses));
  }
  await writeFile(new URL("pipeline.json", output), JSON.stringify(manifest));
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  console.warn("Complete fitted-pose preview requires the cpu-pipeline build documented in web/README.md.");
}

const poseTarget = new URL("../public/pose/", import.meta.url);
await mkdir(poseTarget, { recursive: true });
const fixture = JSON.parse(await readFile(new URL("../../pose-fitter/tests/fixtures/cpu-golden.json", import.meta.url), "utf8"));
await writeFile(new URL("geometry.json", poseTarget), JSON.stringify(fixture.geometry));
try {
  await copyFile(new URL("../../pose-fitter/target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm", import.meta.url), new URL("cube_pose_fitter.wasm", poseTarget));
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  console.warn("Pose overlay requires the WASM build documented in web/README.md.");
}
