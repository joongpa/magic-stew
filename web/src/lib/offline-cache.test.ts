import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {buildOfflineRelease} from '../../scripts/offline-build';
import {installCacheWorker} from '../../scripts/service-worker';
import type {CachedAsset, OfflineRelease, CacheWorkerScope} from '../../scripts/service-worker';
import {offlineCacheReady, startOfflineCache} from './offline-cache';

const digest = (body: string) => createHash('sha256').update(body).digest();
function asset(path: string, body: string, precache = true): CachedAsset {
  return {path, revision: digest(body).toString('hex'), integrity: `sha256-${digest(body).toString('base64')}`, precache};
}
class MemoryCache {
  entries = new Map<string, Response>();
  full = false;
  async match(input: Request | string) { return this.entries.get(typeof input === 'string' ? input : input.url)?.clone(); }
  async put(input: Request | string, response: Response) {
    if (this.full) throw new DOMException('Full', 'QuotaExceededError');
    this.entries.set(typeof input === 'string' ? input : input.url, response.clone());
  }
  async keys() { return [...this.entries.keys()].map(url => new Request(url)); }
  async delete(input: Request | string) { return this.entries.delete(typeof input === 'string' ? input : input.url); }
}
class MemoryStorage {
  stores = new Map<string, MemoryCache>();
  async open(name: string) {
    if (!this.stores.has(name)) this.stores.set(name, new MemoryCache());
    return this.stores.get(name)!;
  }
}
function worker(release: OfflineRelease, bodies: Map<string, string>, storage = new MemoryStorage()) {
  const handlers = new Map<string, (event: any) => void>();
  const downloads: string[] = [];
  let online = true, claims = 0;
  installCacheWorker({
    registration: {scope: 'https://example.test/cube/'},
    caches: storage as unknown as CacheStorage,
    clients: {async claim() { claims++; }},
    async fetch(input: RequestInfo | URL) {
      const request = input as Request;
      if (!online) throw new TypeError('Offline');
      downloads.push(request.url);
      const body = bodies.get(new URL(request.url).pathname.replace('/cube/', ''));
      if (body === undefined) return new Response('Not found', {status: 404});
      if (request.integrity !== `sha256-${digest(body).toString('base64')}`) throw new TypeError('Integrity mismatch');
      return new Response(body, {headers: {'Cross-Origin-Opener-Policy': 'same-origin'}});
    },
    addEventListener(type: string, handler: (event: any) => void) { handlers.set(type, handler); },
  } as CacheWorkerScope, release);
  return {
    downloads, storage, handlers, get claims() { return claims; }, set online(value: boolean) { online = value; },
    async lifecycle(type: 'install' | 'activate') {
      const work: Promise<unknown>[] = [];
      handlers.get(type)!({waitUntil(p: Promise<unknown>) { work.push(p); }});
      await Promise.all(work);
    },
    fetch(path: string, options: {method?: string; mode?: string; range?: boolean} = {}) {
      let result: Promise<Response> | undefined;
      handlers.get('fetch')!({request: {
        url: new URL(path, 'https://example.test/cube/').href,
        method: options.method ?? 'GET', mode: options.mode ?? 'cors',
        headers: new Headers(options.range ? {Range: 'bytes=0-8'} : {}),
      }, respondWith(value: Promise<Response>) { result = value; }});
      return result;
    },
  };
}

test('offline shell, lazy workers and models survive a network outage under a project base path', async () => {
  const bodies = new Map([['index.html', 'app'], ['assets/lazy-worker.js', 'worker'],
    ['pose/pipeline.json', 'pipeline'], ['runtime/engine.wasm', 'wasm'], ['models/pose.hash.onnx', 'weights']]);
  const assets = [...bodies].map(([path, body]) => asset(path, body, !path.endsWith('.onnx')));
  const w = worker({version: 'one', assets}, bodies);
  await w.lifecycle('install');
  assert.equal(w.downloads.length, 4, 'install does not preload optional/unused model weights');
  const responses = await Promise.all([w.fetch('models/pose.hash.onnx')!, w.fetch('models/pose.hash.onnx')!]);
  assert.equal(w.downloads.length, 5, 'concurrent requests share one model download');
  assert.equal(await responses[0].text(), 'weights');
  assert.equal(await responses[1].text(), 'weights');
  w.online = false;
  for (const path of ['', 'timer', 'timer/?from=bookmark']) {
    const response = await w.fetch(path, {mode: 'navigate'})!;
    assert.equal(await response.text(), 'app');
    assert.equal(response.headers.get('Cross-Origin-Opener-Policy'), 'same-origin');
  }
  for (const [path, body] of bodies) assert.equal(await (await w.fetch(path)!).text(), body);
  assert.equal(w.downloads.length, 5);
  assert.equal(w.fetch('__diagnostics/run'), undefined);
  assert.equal(w.fetch('models/pose.hash.onnx', {method: 'POST'}), undefined);
  assert.equal(w.fetch('models/pose.hash.onnx', {range: true}), undefined);
  assert.equal(w.fetch('https://elsewhere.test/cube/index.html'), undefined);
});

test('UI releases reuse model bytes, wait for old clients, and clean only obsolete app assets on activation', async () => {
  const bodies = new Map([['index.html', 'old UI'], ['models/pose.hash.onnx', 'unchanged weights']]);
  const old = worker({version: 'old', assets: [...bodies].map(([p, b]) => asset(p, b, p === 'index.html'))}, bodies);
  await old.lifecycle('install'); await old.fetch('models/pose.hash.onnx');
  const unrelated = await old.storage.open('camera-recordings');
  await unrelated.put('https://example.test/private-recording', new Response('keep'));
  bodies.set('index.html', 'new UI');
  const next = worker({version: 'next', assets: [...bodies].map(([p, b]) => asset(p, b, p === 'index.html'))}, bodies, old.storage);
  await next.lifecycle('install');
  assert.equal(next.claims, 0, 'installation does not take over a running solve');
  assert.equal(next.downloads.length, 1, 'UI update only fetches the changed shell');
  assert.equal(await (await next.fetch('models/pose.hash.onnx')!).text(), 'unchanged weights');
  assert.equal(next.downloads.length, 1);
  assert.equal((await old.storage.open('magic-stew-assets-v1:/cube/')).entries.size, 3, 'old shell remains while update waits');
  await next.lifecycle('activate');
  assert.equal(next.claims, 1);
  assert.equal((await old.storage.open('magic-stew-assets-v1:/cube/')).entries.size, 2);
  assert.equal(unrelated.entries.size, 1, 'recording caches are never cleaned by the asset worker');
  next.online = false;
  assert.equal(await (await next.fetch('timer', {mode: 'navigate'})!).text(), 'new UI');
});

test('failed and mismatched downloads are not cached; a full cache still allows online inference', async () => {
  const entry = asset('models/pose.hash.onnx', 'expected weights', false);
  const bodies = new Map<string, string>();
  const w = worker({version: 'one', assets: [entry]}, bodies);
  await assert.rejects(w.fetch(entry.path)!, /404/);
  bodies.set(entry.path, 'wrong weights');
  await assert.rejects(w.fetch(entry.path)!, /Integrity/);
  const cache = await w.storage.open('magic-stew-assets-v1:/cube/');
  assert.equal(cache.entries.size, 0);
  cache.full = true;
  bodies.set(entry.path, 'expected weights');
  assert.equal(await (await w.fetch(entry.path)!).text(), 'expected weights');
});

async function fixture(directory: string, ui: string) {
  const content: Record<string, string> = {
    'index.html': `<html><head></head><body>${ui}</body></html>`,
    'assets/app.js': ui,
    'runtime/ort.wasm': 'ort',
    'models/detector.json': JSON.stringify({defaultSize: 160, bbox: {file: 'bbox.onnx'}, variants: [
      {size: 160, file: 'unused-missing-160.onnx'}, {size: 256, file: 'pose.onnx'}, {size: 288, file: 'unused-288.onnx'},
    ]}),
    'models/bbox.onnx': 'bbox weights', 'models/pose.onnx': 'pose weights',
    'models/unused-288.onnx': 'unused weights',
    'models/cpu-bench-input.f32': 'unused benchmark',
    'pose/pipeline.json': JSON.stringify({rust: 'rust.current.wasm', geometry: 'geometry.current.json',
      opencvModule: 'opencv.current.mjs', opencvWasm: 'opencv.current.wasm'}),
    'pose/rust.current.wasm': 'rust', 'pose/geometry.current.json': '{}',
    'pose/opencv.current.mjs': 'module', 'pose/opencv.current.wasm': 'opencv',
    'pose/rust.obsolete.wasm': 'obsolete', 'pose/licenses/LICENSE': 'license',
  };
  for (const [path, body] of Object.entries(content)) {
    await mkdir(join(directory, path, '..'), {recursive: true});
    await writeFile(join(directory, path), body);
  }
}

test('production build fingerprints actual model bytes independently of UI releases and includes lazy chunks', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'offline-build-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  await fixture(join(directory, 'old'), 'old UI');
  await fixture(join(directory, 'new'), 'new UI');
  const old = await buildOfflineRelease(join(directory, 'old'));
  const next = await buildOfflineRelease(join(directory, 'new'));
  assert.notEqual(old.version, next.version);
  assert.deepEqual(old.assets.filter(a => a.path.endsWith('.onnx')), next.assets.filter(a => a.path.endsWith('.onnx')));
  assert.ok(next.assets.find(a => a.path === 'assets/app.js')?.precache);
  assert.ok(!next.assets.some(a => a.path.includes('obsolete') || a.path.includes('cpu-bench')));
  const detector = JSON.parse(await readFile(join(directory, 'new/models/detector.json'), 'utf8'));
  assert.equal(detector.defaultSize, 256);
  assert.deepEqual(detector.variants.map((variant: {size: number}) => variant.size), [256]);
  assert.equal(next.assets.filter(asset => asset.path.endsWith('.onnx')).length, 2);
  assert.match(detector.variants[0].file, /^pose\.[a-f0-9]{16}\.onnx$/);
  assert.equal(await readFile(join(directory, 'new/models', detector.variants[0].file), 'utf8'), 'pose weights');
  assert.equal((await readdir(join(directory, 'new/models'))).length, 3, 'no duplicate unversioned model files');
  assert.match(await readFile(join(directory, 'new/index.html'), 'utf8'), new RegExp(next.version));
  const source = await readFile(join(directory, 'new/service-worker.js'), 'utf8');
  assert.doesNotThrow(() => new Function(source), 'worker output is standalone JavaScript');
});

test('registration is scoped, ignores HTTP cache for updates, and degrades to online use on failure', async t => {
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  t.after(() => {
    if (previousNavigator) Object.defineProperty(globalThis, 'navigator', previousNavigator); else Reflect.deleteProperty(globalThis, 'navigator');
    if (previousWindow) Object.defineProperty(globalThis, 'window', previousWindow); else Reflect.deleteProperty(globalThis, 'window');
  });
  Object.defineProperty(globalThis, 'window', {configurable: true, value: {isSecureContext: true}});
  const events = new EventTarget();
  const calls: any[] = [];
  const serviceWorker = Object.assign(events, {
    controller: {scriptURL: 'https://example.test/cube/service-worker.js',
      postMessage(_data: unknown, ports: MessagePort[]) { ports[0].postMessage({version: 'release'}); ports[0].close(); }},
    async register(...args: any[]) { calls.push(args); return {}; },
  });
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {serviceWorker}});
  assert.equal(await startOfflineCache('https://example.test/cube/', 'release'), true);
  assert.equal(await offlineCacheReady(), true);
  assert.deepEqual(calls[0], ['https://example.test/cube/service-worker.js', {scope: '/cube/', updateViaCache: 'none'}]);
  serviceWorker.register = async () => { throw new Error('Storage disabled'); };
  t.mock.method(console, 'warn', () => {});
  assert.equal(await startOfflineCache('https://example.test/cube/', 'release'), false);
});
