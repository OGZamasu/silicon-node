// Silicon Node — Electron shell: real app window + tray, replacing the
// Edge app-mode shortcut and the pystray icon in one process.
"use strict";

const { app, BrowserWindow, Menu, Tray, nativeImage, shell, ipcMain } =
  require("electron");
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { execFile, execFileSync } = require("child_process");
const ICON = path.join(__dirname, "..", "..", "server", "ui", "icon.png");
const RECOVERY_PAGE = path.join(__dirname, "recovery.html");
const PRELOAD = path.join(__dirname, "preload.js");

const NODE = "http://127.0.0.1:8790";
const UI = NODE + "/ui";
// This shell runs on Windows, so everything it sends reaches the WSL
// service through the port proxy — never from loopback — and needs the
// swarm token like any other off-box caller (SECURITY.md). It is read
// from the same file the PySide tray uses; if that fails the page is
// opened bare and asks for the token itself.
const PREFERRED_DISTRO = process.env.SILICON_NODE_WSL_DISTRO?.trim() ||
  "SiliconNode";
let tokenCache = { t: "", at: 0 };   // the share is slow; re-read each minute
let distroCache = { names: null, at: 0 };

function installedWslDistros() {
  if (Date.now() - distroCache.at < 30_000 && distroCache.names)
    return distroCache.names;
  let names = [];
  try {
    names = decodeWslOutput(execFileSync("wsl.exe", ["--list", "--quiet"], {
      encoding: "buffer", timeout: 8000, windowsHide: true,
    })).split(/\r?\n/).map((name) => name.replace(/^\s*\*?\s*/, "").trim())
      .filter(Boolean);
  } catch {}
  distroCache = { names, at: Date.now() };
  return names;
}

function resolveDistro() {
  const names = installedWslDistros();
  if (names.includes(PREFERRED_DISTRO)) return PREFERRED_DISTRO;
  if (!process.env.SILICON_NODE_WSL_DISTRO?.trim() && names.length === 1)
    return names[0];
  return null;
}

function swarmToken() {
  if (Date.now() - tokenCache.at < 60_000) return tokenCache.t;
  const parse = (s) => {
    const t = JSON.parse(s).swarm_token;
    return typeof t === "string" ? t.trim() : "";
  };
  const distro = resolveDistro();
  let t = "";
  if (distro) {
    const swarmJson = `\\\\wsl$\\${distro}\\opt\\silicon\\swarm.json`;
    try { t = parse(fs.readFileSync(swarmJson, "utf8")); } catch {}
  }
  if (!t && distro) {
    // The \\wsl$ share is not always mounted when the tray starts with
    // Windows; ask the distro itself instead.
    try {
      t = parse(execFileSync("wsl.exe",
        ["-d", distro, "--", "cat", "/opt/silicon/swarm.json"],
        { encoding: "utf8", timeout: 8000, windowsHide: true }));
    } catch {}
  }
  tokenCache = { t, at: Date.now() };
  return t;
}
function authed(url) {
  const t = swarmToken();
  return fetch(url, t ? { headers: { Authorization: `Bearer ${t}` } } : {});
}

function isRecoveryPage(event) {
  return event.senderFrame?.url === pathToFileURL(RECOVERY_PAGE).href;
}

async function nodeIsHealthy() {
  try {
    const response = await fetch(`${NODE}/health`, {
      signal: AbortSignal.timeout(2500),
    });
    return response.ok;
  } catch {
    return false;
  }
}

function decodeWslOutput(output) {
  if (!Buffer.isBuffer(output)) return String(output || "").trim();
  if (!output.length) return "";

  let text;
  if (output[0] === 0xff && output[1] === 0xfe) {
    text = output.subarray(2).toString("utf16le");
  } else if (output[0] === 0xfe && output[1] === 0xff) {
    const bytes = Buffer.from(output.subarray(2));
    for (let i = 0; i + 1 < bytes.length; i += 2) {
      const byte = bytes[i];
      bytes[i] = bytes[i + 1];
      bytes[i + 1] = byte;
    }
    text = bytes.toString("utf16le");
  } else {
    let oddNuls = 0;
    let evenNuls = 0;
    for (let i = 0; i < output.length; i++) {
      if (output[i] === 0) {
        if (i % 2) oddNuls++;
        else evenNuls++;
      }
    }
    if (oddNuls > output.length / 8) {
      text = output.toString("utf16le");
    } else if (evenNuls > output.length / 8) {
      const bytes = Buffer.from(output);
      for (let i = 0; i + 1 < bytes.length; i += 2) {
        const byte = bytes[i];
        bytes[i] = bytes[i + 1];
        bytes[i + 1] = byte;
      }
      text = bytes.toString("utf16le");
    } else {
      text = output.toString("utf8");
    }
  }

  return text.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "")
    .trim();
}

function startWslService() {
  return new Promise((resolve) => {
    const distro = resolveDistro();
    if (!distro) {
      const names = installedWslDistros();
      let message;
      if (!names.length) {
        message = "No WSL distro was found. Install and provision a WSL distro, then try again.";
      } else if (process.env.SILICON_NODE_WSL_DISTRO?.trim()) {
        message = `The configured WSL distro “${PREFERRED_DISTRO}” was not found. Check SILICON_NODE_WSL_DISTRO.`;
      } else {
        message = `More than one WSL distro is installed. Set SILICON_NODE_WSL_DISTRO to the one that contains silicon-node.service (${names.join(", ")}).`;
      }
      resolve({ ok: false, message, detail: names.join("\n") });
      return;
    }

    execFile("wsl.exe", ["-d", distro, "-u", "root", "--",
      "systemctl", "start", "silicon-node.service"], {
      windowsHide: true,
      timeout: 120_000,
      maxBuffer: 64 * 1024,
      encoding: "buffer",
    }, (error, stdout, stderr) => {
      if (!error) {
        resolve({ ok: true });
        return;
      }

      const detail = decodeWslOutput(stderr) || decodeWslOutput(stdout) ||
        error.message;
      let message;
      if (error.code === "ENOENT") {
        message = "WSL was not found. Install or enable WSL, then try again.";
      } else if (/no distribution|distribution .* not found/i.test(detail)) {
        message = `The WSL distro “${distro}” was not found. Check the distro name and setup, then try again.`;
      } else if (/not been booted with systemd|systemd is not running/i.test(detail)) {
        message = `systemd is not enabled in “${distro}”. Enable it in /etc/wsl.conf, restart WSL, and try again.`;
      } else if (/unit .* (?:could not be found|not found)|not-found/i.test(detail)) {
        message = `The silicon-node service is not installed in “${distro}”. Finish the service setup in the provisioning guide, then try again.`;
      } else if (error.killed) {
        message = `Starting the service in “${distro}” timed out. Check that the distro can start, then try again.`;
      } else {
        message = `WSL could not start silicon-node.service in “${distro}”. Check that the distro is registered, systemd is enabled, and the service is installed.`;
      }
      resolve({ ok: false, message, detail: detail.slice(0, 1200) });
    });
  });
}

ipcMain.handle("node:check", (event) =>
  isRecoveryPage(event) ? nodeIsHealthy() : false);
ipcMain.handle("node:start-services", (event) =>
  isRecoveryPage(event) ? startWslService() :
    { ok: false, message: "This action is only available from the recovery screen." });
ipcMain.on("node:open-dashboard", (event) => {
  if (!isRecoveryPage(event) || !win || win.isDestroyed()) return;
  const t = swarmToken();
  win.loadURL(t ? `${UI}#token=${encodeURIComponent(t)}` : UI)
    .catch((error) => console.error("Could not open Silicon Node UI:", error));
});
// The owner's project hub, if they keep one: an environment setting, since
// this repository is public. The menu item appears only when it is set.
const HUB = process.env.SILICON_NODE_HUB_URL || "";
const COLORS = { idle: "#34c759", job: "#ff9500", llm: "#0a84ff",
                 down: "#ff453a" };

if (!app.requestSingleInstanceLock()) app.quit();

let win = null;
let tray = null;

// A silicon-die glyph drawn as a data-URL PNG per status color.
function glyph(color) {
  const c = { "#34c759": [52, 199, 89], "#ff9500": [255, 149, 0],
              "#0a84ff": [10, 132, 255], "#ff453a": [255, 69, 58] }[color];
  const s = 32, px = Buffer.alloc(s * s * 4);
  const set = (x, y, r, g, b, a) => {
    const i = (y * s + x) * 4;
    px[i] = b; px[i + 1] = g; px[i + 2] = r; px[i + 3] = a;
  };
  for (let y = 0; y < s; y++)
    for (let x = 0; x < s; x++) {
      const inDie = x >= 7 && x < 25 && y >= 7 && y < 25;
      const pin = ((x >= 10 && x < 12) || (x >= 15 && x < 17) ||
                   (x >= 20 && x < 12 + 10)) &&
                  ((y >= 2 && y < 7) || (y >= 25 && y < 30));
      const pinH = ((y >= 10 && y < 12) || (y >= 15 && y < 17) ||
                    (y >= 20 && y < 22)) &&
                   ((x >= 2 && x < 7) || (x >= 25 && x < 30));
      if (inDie) set(x, y, c[0], c[1], c[2], 255);
      else if (pin || pinH) set(x, y, 110, 120, 116, 255);
    }
  return nativeImage.createFromBuffer(px, { width: s, height: s });
}

function createWindow() {
  if (win && !win.isDestroyed()) { win.show(); win.focus(); return; }
  win = new BrowserWindow({
    width: 1100, height: 760, minWidth: 880, minHeight: 560,
    title: "Silicon Node",
    icon: ICON,
    backgroundColor: "#1e1e20",
    autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, preload: PRELOAD },
  });
  // Always render a useful local screen first. If the WSL service is down,
  // this gives the user a way to start it instead of an empty Chromium view.
  win.loadFile(RECOVERY_PAGE);
  // External links go to the real browser, not new app windows.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith("http://127.0.0.1") &&
        !url.startsWith("http://localhost")) {
      shell.openExternal(url);
      return { action: "deny" };
    }
    return { action: "allow" };
  });
  win.on("close", (e) => {          // closing hides; the tray owns quit
    if (!app.isQuitting) { e.preventDefault(); win.hide(); }
  });
}

async function pollStatus() {
  try {
    const [health, node, llm] = await Promise.all([
      fetch(`${NODE}/health`).then(r => r.json()),
      authed(`${NODE}/v1/node`).then(r => r.json()),
      authed(`${NODE}/v1/llm`).then(r => r.json()),
    ]);
    const free = node.metrics?.headroom_gb ?? "?";
    if ((health.queue_depth ?? 0) > 0)
      return { key: "job", tip: "Silicon Node — GPU rendering" };
    if (llm.running)
      return { key: "llm",
               tip: `Silicon Node — ${llm.model} loaded · ${free} GB free` };
    return { key: "idle", tip: `Silicon Node — idle · ${free} GB free` };
  } catch {
    return { key: "down", tip: "Silicon Node — service unreachable" };
  }
}

app.setName("Silicon Node");
app.setAppUserModelId("SiliconNode");
app.whenReady().then(() => {
  tray = new Tray(glyph(COLORS.down));
  tray.setToolTip("Silicon Node");
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: "Open Silicon Node", click: createWindow },
    { label: "Open Memories hub", visible: Boolean(HUB),
      click: () => shell.openExternal(HUB) },
    { type: "separator" },
    { label: "Start when I sign in", type: "checkbox",
      checked: app.getLoginItemSettings().openAtLogin,
      click: (item) =>
        app.setLoginItemSettings({ openAtLogin: item.checked }) },
    { type: "separator" },
    { label: "Quit", click: () => { app.isQuitting = true; app.quit(); } },
  ]));
  tray.on("click", createWindow);
  setInterval(async () => {
    const { key, tip } = await pollStatus();
    tray.setImage(glyph(COLORS[key]));
    tray.setToolTip(tip);
  }, 5000);
  createWindow();
});

app.on("second-instance", createWindow);
app.on("window-all-closed", (e) => e.preventDefault());
