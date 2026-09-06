// ShasthoSathi store — IndexedDB wrapper + settings fallback (localStorage).
// Offline-first: all reads/writes are local. Export/import JSON for backup.
const DB_NAME = "adalatai";
const DB_VERSION = 1;
const STORES = ["cases", "audit"];

export function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      for (const s of STORES) {
        if (!db.objectStoreNames.contains(s)) {
          db.createObjectStore(s, { keyPath: "id", autoIncrement: true });
        }
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(store, mode, fn) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const out = fn(t.objectStore(store));
    t.oncomplete = () => resolve(out && out.result !== undefined ? out.result : out);
    t.onerror = () => reject(t.error);
  });
}

export async function add(store, obj) {
  return tx(store, "readwrite", (os) => os.add(obj));
}
export async function all(store) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(store, "readonly").objectStore(store).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
export async function del(store, id) {
  return tx(store, "readwrite", (os) => os.delete(id));
}
export async function put(store, obj) {
  return tx(store, "readwrite", (os) => os.put(obj));
}

export async function exportAll() {
  const dump = { exported: new Date().toISOString(), cases: await all("cases"),
                 audit: await all("audit") };
  return JSON.stringify(dump, null, 1);
}

export async function importAll(jsonText) {
  const data = JSON.parse(jsonText);
  for (const s of STORES) {
    for (const item of data[s] || []) {
      const { id, ...rest } = item;
      await add(s, rest);
    }
  }
  return true;
}

// settings (lang) — localStorage is fine for a single preference
export const settings = {
  get lang() { try { return localStorage.getItem("adalat_lang") || "bn"; } catch { return "bn"; } },
  set lang(v) { try { localStorage.setItem("adalat_lang", v); } catch { /* private mode */ } },
};
