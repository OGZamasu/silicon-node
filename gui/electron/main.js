// Silicon Node — Electron shell: real app window + tray, replacing the
// Edge app-mode shortcut and the pystray icon in one process.
"use strict";

const { app, BrowserWindow, Menu, Tray, nativeImage, shell } =
  require("electron");
const path = require("path");
const ICON = "F:/Windows Silicon Optimizer/silicon-node/server/ui/icon.png";

const UI = "http://127.0.0.1:8790/ui";
const HUB = "https://memories.zamasu.dev/p/silicon-node";
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
    webPreferences: { contextIsolation: true },
  });
  win.loadURL(UI);
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
      fetch("http://127.0.0.1:8790/health").then(r => r.json()),
      fetch("http://127.0.0.1:8790/v1/node").then(r => r.json()),
      fetch("http://127.0.0.1:8790/v1/llm").then(r => r.json()),
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
    { label: "Open Memories hub", click: () => shell.openExternal(HUB) },
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
