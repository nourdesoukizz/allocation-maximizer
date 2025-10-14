import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Simple version checker: compares /version.json with stored value and reloads once if changed
async function checkVersionAndReloadIfNeeded() {
  try {
    const res = await fetch('/version.json', { cache: 'no-store' });
    if (!res.ok) return;
    const data = (await res.json()) as { version?: string };
    const version = data?.version;
    if (!version) return;

    const KEY = 'scx_version';
    const RELOADED_KEY = 'scx_version_reloaded';
    const prev = localStorage.getItem(KEY);

    if (prev && prev !== version) {
      localStorage.setItem(KEY, version);
      const alreadyReloadedFor = localStorage.getItem(RELOADED_KEY);
      if (alreadyReloadedFor !== version) {
        localStorage.setItem(RELOADED_KEY, version);
        // Force a single hard reload to pick up new hashed assets
        window.location.reload();
      }
      return;
    }

    if (prev !== version) localStorage.setItem(KEY, version);
  } catch (_) {
    // silently ignore
  }
}

// Run once on startup and periodically (5 minutes) to detect new deploys
checkVersionAndReloadIfNeeded();
setInterval(checkVersionAndReloadIfNeeded, 5 * 60 * 1000);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
