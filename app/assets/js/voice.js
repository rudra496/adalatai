// ShasthoSathi voice — Web Speech API (bn-BD). Progressive enhancement:
// silently unavailable offline / on unsupported browsers; UI always offers taps.
export function speechSupported() {
  return typeof window !== "undefined" && "webkitSpeechRecognition" in window;
}

export function listen({ lang = "bn-BD", onResult, onError, onEnd }) {
  if (!speechSupported()) {
    onError && onError("unsupported");
    return null;
  }
  const R = window.webkitSpeechRecognition || window.SpeechRecognition;
  const rec = new R();
  rec.lang = lang;
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  rec.onresult = (e) => onResult && onResult(e.results[0][0].transcript);
  rec.onerror = (e) => onError && onError(e.error);
  rec.onend = () => onEnd && onEnd();
  rec.start();
  return rec;
}

export function speak(text, lang = "bn-BD") {
  if (typeof speechSynthesis === "undefined") return false;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang;
  speechSynthesis.speak(u);
  return true;
}
