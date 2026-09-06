// AdalatAI shell — header/nav/footer, audit-trail helper, SW register.
import { t, STRINGS } from "./i18n.js";
import { settings } from "./store.js";

export function applyI18n() {
  const lang = settings.lang || "bn";
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n, lang); });
  const btn = document.getElementById("langToggle");
  if (btn) btn.textContent = t("lang_toggle", lang);
}

const NAV = [
  ["index.html", "home"], ["intake.html", "intake"], ["judge.html", "judge"],
  ["draft.html", "draft"], ["law.html", "law"], ["analytics.html", "analytics"],
  ["audit.html", "audit"],
];

export function initShell(active) {
  const lang = settings.lang || "bn";
  const links = NAV.map(([href, key]) =>
    `<a href="${href}" class="${active === key ? "active" : ""}">${t(key, lang)}</a>`).join("");
  const shell = document.getElementById("shell");
  if (shell) {
    shell.insertAdjacentHTML("afterbegin", `
      <header class="topbar">
        <div class="brand">
          <span class="logo">⚖</span>
          <div>
            <div class="name">AdalatAI <span style="font-weight:400">আদালতএআই</span></div>
            <div class="tag">${t("app_tag", lang)}</div>
          </div>
          <button id="langToggle" class="langbtn" aria-label="Switch language / ভাষা বদলান"></button>
        </div>
        <nav class="nav">${links}</nav>
        <div id="netBanner" class="netbanner"></div>
      </header>`);
  }
  applyI18n();
  const btn = document.getElementById("langToggle");
  if (btn) btn.onclick = () => { settings.lang = (settings.lang === "bn") ? "en" : "bn"; location.reload(); };
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
}

export function el(id) { return document.getElementById(id); }

/** AUDIT TRAIL — every AI action and every judge decision is logged (append-only). */
export async function audit(event, detail) {
  const { add } = await import("./store.js");
  return add("audit", { event, detail: typeof detail === "string" ? detail : JSON.stringify(detail).slice(0, 2000),
                        at: new Date().toISOString(), model_version: detail?.model_version || "case_model-v1" });
}
