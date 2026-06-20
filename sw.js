/* CountGD service worker — offline app shell + cached OpenCV runtime */
const CACHE = 'countgd-v3';
const SHELL = [
  './',
  './index.html',
  './app.js',
  './manifest.json',
  './icon.svg',
];
const CV_URL = 'https://docs.opencv.org/4.9.0/opencv.js';

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL).catch(() => {})));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;

  // OpenCV.js: cache-first (large, immutable) so the app works offline after first load
  if (req.url.startsWith(CV_URL)) {
    e.respondWith(
      caches.open(CACHE).then(async (c) => {
        const hit = await c.match(req);
        if (hit) return hit;
        const res = await fetch(req);
        if (res.ok) c.put(req, res.clone());
        return res;
      })
    );
    return;
  }

  // App shell: network-first with cache fallback (so updates land, offline still works)
  if (new URL(req.url).origin === location.origin) {
    e.respondWith(
      fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => caches.match(req).then((m) => m || caches.match('./index.html')))
    );
  }
});
