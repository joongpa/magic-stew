export interface CachedAsset {
  path: string;
  revision: string;
  integrity: string;
  precache: boolean;
}

export interface OfflineRelease {
  version: string;
  assets: CachedAsset[];
}

export interface CacheWorkerScope {
  registration: {scope: string};
  caches: CacheStorage;
  clients: {claim(): Promise<void>};
  fetch: typeof fetch;
  addEventListener(type: string, handler: (event: any) => void): void;
}

// This function is serialized into the production worker. Keep it self-contained.
export function installCacheWorker(scope: CacheWorkerScope, release: OfflineRelease) {
  const base = new URL(scope.registration.scope);
  const cacheName = `magic-stew-assets-v1:${base.pathname}`;
  const entries = new Map(release.assets.map(asset => [new URL(asset.path, base).pathname, asset]));
  const pending = new Map<string, Promise<Response>>();
  const keyFor = (asset: CachedAsset) => {
    const key = new URL(asset.path, base);
    key.searchParams.set('__revision', asset.revision);
    return key.href;
  };

  async function cached(asset: CachedAsset): Promise<Response> {
    const key = keyFor(asset);
    let cache: Cache;
    try { cache = await scope.caches.open(cacheName); }
    catch {
      return scope.fetch(new Request(new URL(asset.path, base), {
        cache: 'reload', integrity: asset.integrity, credentials: 'same-origin',
      }));
    }
    const hit = await cache.match(key);
    if (hit) return hit;
    let work = pending.get(key);
    if (!work) {
      work = (async () => {
        // Integrity prevents a deployment race or an HTML fallback from poisoning
        // a release's cache. Reload also bypasses stale ordinary HTTP cache entries.
        const response = await scope.fetch(new Request(new URL(asset.path, base), {
          cache: 'reload', integrity: asset.integrity, credentials: 'same-origin',
        }));
        if (!response.ok) throw new Error(`Offline asset download failed: ${asset.path} (${response.status})`);
        try { await cache.put(key, response.clone()); }
        catch { /* Full/disabled storage must not prevent online use. */ }
        return response;
      })();
      pending.set(key, work);
    }
    try { return (await work).clone(); }
    finally { if (pending.get(key) === work) pending.delete(key); }
  }

  scope.addEventListener('install', event => {
    // Precache the shell, all lazy JS chunks, manifests and current WASM runtime.
    // ONNX weights are cached only when requested by the application.
    event.waitUntil((async () => {
      for (const asset of release.assets.filter(asset => asset.precache)) {
        await cached(asset);
      }
    })());
    // Let updates wait for existing tabs to close; never replace a running solve.
  });

  scope.addEventListener('activate', event => {
    event.waitUntil((async () => {
      const cache = await scope.caches.open(cacheName);
      const keep = new Set(release.assets.map(keyFor));
      for (const request of await cache.keys()) {
        if (!keep.has(request.url)) await cache.delete(request);
      }
      await scope.clients.claim();
    })());
  });

  scope.addEventListener('message', event => {
    if (event.data?.type === 'CACHE_VERSION') {
      event.ports?.[0]?.postMessage({version: release.version});
    }
  });

  scope.addEventListener('fetch', event => {
    const request = event.request as Request;
    const url = new URL(request.url);
    if (request.method !== 'GET' || request.headers.has('range') || url.origin !== base.origin) return;
    const root = base.pathname.replace(/\/$/, '');
    const navigation = request.mode === 'navigate' &&
      [base.pathname, `${root}/index.html`, `${root}/timer`, `${root}/timer/`].includes(url.pathname);
    const asset = entries.get(navigation ? `${root}/index.html` : url.pathname);
    if (!asset) return; // Never cache recordings, diagnostics, uploads or unrelated requests.
    event.respondWith(cached(asset));
  });
}
