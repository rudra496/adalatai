// AdalatAI service worker — offline-first, network-first for same-origin (fresh deploys).
const CACHE = "adalatai-v1";
const ASSETS = ["./", "./index.html", "./intake.html", "./judge.html", "./draft.html",
  "./law.html", "./analytics.html", "./audit.html",
  "./assets/css/app.css", "./assets/js/app.js", "./assets/js/court.js",
  "./assets/js/i18n.js", "./assets/js/store.js", "./assets/js/voice.js",
  "./manifest.webmanifest", "./data/case_model.json", "./data/penal_code_full.json"];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin === location.origin) {
    e.respondWith(fetch(e.request, { cache: "no-cache" }).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
      return res;
    }).catch(() => caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || Response.error())));
    return;
  }
  e.respondWith(fetch(e.request).then((res) => { const c = res.clone(); caches.open(CACHE).then((x) => x.put(e.request, c)); return res; }).catch(() => caches.match(e.request)));
});
