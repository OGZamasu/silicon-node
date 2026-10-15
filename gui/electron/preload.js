"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("siliconNodeRecovery", {
  check: () => ipcRenderer.invoke("node:check"),
  startServices: () => ipcRenderer.invoke("node:start-services"),
  openDashboard: () => ipcRenderer.send("node:open-dashboard"),
});
