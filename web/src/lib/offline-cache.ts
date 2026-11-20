let readiness: Promise<boolean> = Promise.resolve(false);

/** Install before model initialization, without blocking the application shell. */
export function startOfflineCache(baseUrl: string, release: string | undefined) {
  readiness = registerCache(baseUrl, release).catch(error => {
    console.warn('Offline caching is unavailable; continuing online.', error);
    return false;
  });
  return readiness;
}

export function offlineCacheReady() { return readiness; }

async function registerCache(baseUrl: string, release: string | undefined): Promise<boolean> {
  if (!('serviceWorker' in navigator) || !window.isSecureContext) return false;
  const script = new URL('service-worker.js', baseUrl);
  const registration = await navigator.serviceWorker.register(script.href, {
    scope: new URL(baseUrl).pathname, updateViaCache: 'none',
  });
  const controlled = () => navigator.serviceWorker.controller?.scriptURL === script.href;
  if (!controlled()) {
    const ready = await new Promise<boolean>(resolve => {
      const finish = (success: boolean) => {
        clearTimeout(timeout);
        navigator.serviceWorker.removeEventListener('controllerchange', changed);
        installing?.removeEventListener('statechange', failed);
        resolve(success);
      };
      const changed = () => { if (controlled()) finish(true); };
      const installing = registration.installing;
      const failed = () => { if (installing?.state === 'redundant') finish(false); };
      const timeout = setTimeout(() => finish(false), 30_000);
      navigator.serviceWorker.addEventListener('controllerchange', changed);
      installing?.addEventListener('statechange', failed);
      if (controlled()) finish(true);
      else failed();
    });
    if (!ready) return false;
  }
  // If deployment changed between the initial HTML and first worker install,
  // reload through the worker's pinned shell before opening model sessions.
  if (release) {
    const workerVersion = await new Promise<string | undefined>(resolve => {
      const channel = new MessageChannel();
      const timeout = setTimeout(() => { channel.port1.close(); resolve(undefined); }, 3000);
      channel.port1.onmessage = event => {
        clearTimeout(timeout); channel.port1.close(); resolve(event.data?.version);
      };
      navigator.serviceWorker.controller!.postMessage({type: 'CACHE_VERSION'}, [channel.port2]);
    });
    if (workerVersion && workerVersion !== release) {
      window.location.reload();
      await new Promise<never>(() => {});
    }
  }
  return true;
}
