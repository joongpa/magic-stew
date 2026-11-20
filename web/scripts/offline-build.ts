import {createHash} from 'node:crypto';
import {mkdir, readFile, readdir, rename, rm, writeFile} from 'node:fs/promises';
import {dirname, join, posix, resolve} from 'node:path';
import {transformWithEsbuild} from 'vite';
import type {Plugin, ResolvedConfig} from 'vite';
import type {CachedAsset, OfflineRelease} from './service-worker';

const sha256 = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex');
const safePath = (path: string) => {
  if (!path || path.startsWith('/') || path.includes('\\') || path.split('/').includes('..') || /[?#]/.test(path)) {
    throw new Error(`Expected a relative static asset path: ${path}`);
  }
  return path;
};

async function files(directory: string, prefix = ''): Promise<string[]> {
  const result: string[] = [];
  for (const item of await readdir(join(directory, prefix), {withFileTypes: true})) {
    const path = posix.join(prefix, item.name);
    if (item.isDirectory()) result.push(...await files(directory, path));
    else if (item.isFile()) result.push(path);
  }
  return result.sort();
}

/** Operates only on build output; source manifests and local training files stay unchanged. */
export async function buildOfflineRelease(output: string): Promise<OfflineRelease> {
  const modelPath = join(output, 'models/detector.json');
  const detector = JSON.parse(await readFile(modelPath, 'utf8'));
  const pose = detector.variants.find((variant: {size: number}) => variant.size === 256);
  if (!pose) throw new Error('The production detector requires a 256px pose model.');
  detector.defaultSize = 256;
  detector.variants = [pose];
  const models = new Map<string, string>();
  for (const variant of [detector.bbox, ...detector.variants]) {
    const original = safePath(variant.file);
    if (!models.has(original)) {
      const bytes = await readFile(join(output, 'models', original));
      const digest = sha256(bytes);
      const versioned = original.replace(/\.onnx$/, `.${digest.slice(0, 16)}.onnx`);
      if (versioned === original) throw new Error(`Expected an ONNX model: ${original}`);
      await mkdir(dirname(join(output, 'models', versioned)), {recursive: true});
      await rename(join(output, 'models', original), join(output, 'models', versioned));
      models.set(original, versioned);
    }
    variant.file = models.get(original);
  }
  await writeFile(modelPath, JSON.stringify(detector, null, 2) + '\n');

  const pipeline = JSON.parse(await readFile(join(output, 'pose/pipeline.json'), 'utf8'));
  const poseFiles = new Set(['pipeline.json', ...['rust', 'geometry', 'opencvModule', 'opencvWasm'].map(key => safePath(pipeline[key]))]);
  for (const file of poseFiles) await readFile(join(output, 'pose', file));
  // sync:runtime accumulates historical hashes locally. Ship only the current pipeline.
  for (const path of await files(join(output, 'pose'))) {
    if (!poseFiles.has(path) && !path.startsWith('licenses/')) await rm(join(output, 'pose', path));
  }
  const modelFiles = new Set(['detector.json', ...models.values()]);
  for (const path of await files(join(output, 'models'))) {
    if (!modelFiles.has(path)) await rm(join(output, 'models', path));
  }

  const assets: CachedAsset[] = [];
  for (const path of await files(output)) {
    if (['_headers', '_redirects', '.nojekyll', 'timer/index.html', 'service-worker.js', 'offline-release.json', '404.html'].includes(path)) continue;
    const bytes = await readFile(join(output, path));
    const revision = sha256(bytes);
    assets.push({path, revision, integrity: `sha256-${Buffer.from(revision, 'hex').toString('base64')}`,
      precache: !path.endsWith('.onnx') && !path.startsWith('pose/licenses/')});
  }
  const worker = await transformWithEsbuild(await readFile(new URL('./service-worker.ts', import.meta.url), 'utf8'),
    'service-worker.ts', {format: 'iife', globalName: 'OfflineCache', keepNames: false, minify: true});
  const version = sha256(JSON.stringify(assets) + worker.code).slice(0, 20);
  const indexPath = join(output, 'index.html');
  const html = (await readFile(indexPath, 'utf8')).replace('</head>',
    `  <meta name="app-release" content="${version}" />\n  </head>`);
  await writeFile(indexPath, html);
  // Static hosts can open the timer directly without an SPA rewrite rule.
  await mkdir(join(output, 'timer'), {recursive: true});
  await writeFile(join(output, 'timer/index.html'), html);
  await writeFile(join(output, '.nojekyll'), '');
  const index = assets.find(asset => asset.path === 'index.html')!;
  index.revision = sha256(html);
  index.integrity = `sha256-${Buffer.from(index.revision, 'hex').toString('base64')}`;
  const release = {version, assets};
  await writeFile(join(output, 'offline-release.json'), JSON.stringify(release, null, 2) + '\n');
  await writeFile(join(output, 'service-worker.js'), `${worker.code}\nOfflineCache.installCacheWorker(self, ${JSON.stringify(release)});\n`);
  return release;
}

export function offlineBuild(): Plugin {
  let config: ResolvedConfig;
  return {
    name: 'magic-stew-offline', apply: 'build',
    configResolved(value) { config = value; },
    async closeBundle() {
      const release = await buildOfflineRelease(resolve(config.root, config.build.outDir));
      console.log(`Offline release ${release.version}: ${release.assets.length} versioned assets; models cached on demand.`);
    },
  };
}
