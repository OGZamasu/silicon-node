"""DeepSeek Harness (dsh) manager — the agentic chat sidecar.

Port of the Mac's HarnessRuntime: same pinned package, same isolated
DSH_HOME, same provider mechanics (openai-completions provider pointed at
the locally served model, contextWindow advertised so the harness budgets
compaction against reality). Adapted for this node:

- dsh runs on native Windows (spawned through WSL interop) because the
  model listens on Windows loopback.
- The provider's model id is the *real* engine id — ninfer validates the
  model field where llama-server ignores it.
- Node.js is a private portable runtime beside the app (v22, above the
  harness's 20.12 floor), never the machine's PATH node.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .llm import LLM, PORT as LLM_PORT

log = logging.getLogger("silicon-node.harness")

PACKAGE_SPEC = "@deepseek-ai/dsh@0.1.0-rc.7"   # pinned, like the Mac
# DSH's caret ranges admit later prerelease components. Keep the rc.7 bundle
# consistent instead of mixing it with the separately published rc.8 train.
PACKAGE_NPM_BEFORE = "2026-08-18T00:00:00Z"
PROVIDER_ID = "silicon-local"
API_KEY_VAR = "SILICON_LOCAL_API_KEY"
WEB_PORT = int(os.environ.get("SILICON_NODE_DSH_PORT", "8090"))

from . import hostos  # noqa: E402 (after the constants it reads)

_RUNTIME = Path(os.environ.get(
    "SILICON_NODE_RUNTIME",
    str(hostos.WIN_RUNTIME)
    if hostos.IS_WSL
    else str(Path(__file__).resolve().parent.parent / "runtime")))
NODE_EXE = _RUNTIME / "node" / ("node.exe" if hostos.IS_WSL
                                else "bin/node")
DSH_PKG = _RUNTIME / "dsh" / "node_modules" / "@deepseek-ai" / "dsh"
DSH_HOME_WSL = _RUNTIME / "dsh-home"
NODE_RELEASE_INDEX = "https://nodejs.org/dist/index.json"
NODE_RELEASE_BASE = "https://nodejs.org/dist"
MAX_NODE_ARCHIVE_BYTES = 150 * 1024 * 1024


def _engine_path(p: Path) -> str:
    """A path as the NODE PROCESS must see it — Windows form through
    interop on WSL, POSIX on Linux."""
    return hostos.win_path(p) if hostos.IS_WSL else str(p)


def _dsh_entry(package_dir: Path = DSH_PKG) -> Path | None:
    """Find the installed package's bin script, tolerating partial installs."""
    try:
        data = json.loads((package_dir / "package.json").read_text())
        bin_field = data.get("bin")
        rel = bin_field if isinstance(bin_field, str) else \
            next(iter(bin_field.values()), None) if bin_field else None
        entry = package_dir / rel if rel else None
        return entry if entry and entry.is_file() else None
    except (OSError, ValueError, TypeError, StopIteration):
        return None


def _dsh_entry_win() -> str | None:
    """The package bin script as the node process must see it."""
    entry = _dsh_entry()
    if not entry:
        return None
    return hostos.win_path(entry) if hostos.IS_WSL else str(entry)


def _missing_peer_specs(package_root: Path) -> list[str]:
    """Return missing required npm peers as direct install specifications.

    DSH publishes its modules with a large peer-dependency graph. Resolving
    the entire graph in npm can take an excessive amount of time, so install
    the package graph first with legacy peer resolution, then add only the
    peer packages that Node cannot resolve from their consumers.
    """
    node_modules = package_root / "node_modules"
    if not node_modules.is_dir():
        return []

    missing: dict[str, list[str]] = {}
    for manifest in sorted(node_modules.rglob("package.json")):
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError, TypeError):
            continue
        peers = data.get("peerDependencies")
        if not isinstance(peers, dict):
            continue
        optional = data.get("peerDependenciesMeta") or {}
        for name, version_range in peers.items():
            if (optional.get(name, {}).get("optional")
                    or not isinstance(version_range, str)):
                continue
            peer_path = Path(*name.split("/"))
            owner = manifest.parent
            found = False
            while True:
                if (owner / "node_modules" / peer_path / "package.json").is_file():
                    found = True
                    break
                if owner == package_root or package_root not in owner.parents:
                    break
                owner = owner.parent
            if not found:
                missing.setdefault(name, []).append(version_range)

    # These UI peers have a common range across the supported DSH packages.
    # Version 18 satisfies packages that accept React 16–18 and those that
    # also accept 19.
    overrides = {"react": "^18.0.0", "react-dom": "^18.0.0"}
    specs = []
    for name, ranges in sorted(missing.items()):
        version_range = overrides.get(name, ranges[0])
        specs.append(f"{name}@{version_range}")
    return specs


def _archive_target(root: Path, name: str) -> Path:
    """Resolve an archive member below root, rejecting path traversal."""
    rel = PurePosixPath(name.replace("\\", "/"))
    if (rel.is_absolute() or ".." in rel.parts
            or (rel.parts and rel.parts[0].endswith(":"))):
        raise RuntimeError("The Node.js archive contains an unsafe path.")
    target = (root / Path(*rel.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RuntimeError("The Node.js archive contains an unsafe path.")
    return target


def _safe_extract_zip(archive: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in archive.infolist():
        _archive_target(root, member.filename)
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError("The Node.js ZIP contains a symbolic link.")
    archive.extractall(target)


def _safe_extract_tar(archive: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    for member in archive.getmembers():
        _archive_target(root, member.name)
        if member.issym():
            link = PurePosixPath(member.linkname.replace("\\", "/"))
            member_path = PurePosixPath(member.name.replace("\\", "/"))
            base = root / Path(*member_path.parts[:-1])
            resolved_link = (base / Path(*link.parts)).resolve()
            if not resolved_link.is_relative_to(root):
                raise RuntimeError("The Node.js archive contains an unsafe link.")
        elif member.islnk():
            _archive_target(root, member.linkname)
        elif not (member.isfile() or member.isdir()):
            raise RuntimeError(
                "The Node.js archive contains an unsupported file type.")
    archive.extractall(target)


def _settings_yaml(model_id: str, model_name: str, context: int,
                   port: int) -> str:
    return f"""# Managed by Silicon Node: the '{PROVIDER_ID}' provider below is how the
# harness reaches the model this node serves locally.
llm-pi-ai:
  providers:
    {PROVIDER_ID}:
      apiKeyEnv: {API_KEY_VAR}
      api: openai-completions
      baseURL: http://127.0.0.1:{port}/v1
      models:
        - id: {json.dumps(model_id)}
          name: {json.dumps(model_name)}
          contextWindow: {context}
ui-onboarding:
  welcomeNoticeVersion: 2026-08-13.1
agent-default-model:
  provider: {PROVIDER_ID}
  model: {json.dumps(model_id)}
"""


def _local_only_patch_yaml() -> str:
    """Disable DSH's hosted DeepSeek model adapter.

    Silicon Node supplies the active local engine through ``llm-pi-ai``.
    The DSH web bundle mounts a hosted DeepSeek adapter by default, so its
    catalog would otherwise remain visible beside the local model.
    """
    return """# Managed by Silicon Node: expose only the active local model.
- id: llm-deepseek
  disabled: true
"""


def _active_model_settings() -> tuple[str, str, int, int]:
    """Return the model endpoint that is actually serving on this node.

    The engines bind Windows loopback and DSH runs as a native Windows
    process, so it can reach the active engine directly. Prefer engines in
    the same order as the node's /v1/chat/completions proxy.
    """
    if LLM.running:
        model_id = getattr(LLM, "model_id", "qwen3.8-27b") or "qwen3.8-27b"
        context = 65536 if LLM._profile == "c1" else 32768
        return model_id, f"{model_id} (this PC)", LLM_PORT, context

    from .llamacpp import LLAMACPP, PORT as GGUF_PORT  # noqa: PLC0415
    gguf = LLAMACPP.status()
    if gguf["running"]:
        model_id = gguf.get("model")
        context = gguf.get("context_length")
        if not model_id or not context:
            # A separately invoked manager can see the live port, but its
            # in-memory model metadata belongs to the main server process.
            # Ask llama-server for its actual loaded model and trained ctx.
            try:
                inventory = json.loads(hostos.http_get(
                    f"http://127.0.0.1:{GGUF_PORT}/v1/models", timeout=4))
                items = inventory.get("data") or inventory.get("models") or []
                if items:
                    item = items[0]
                    model_id = model_id or item.get("id") or item.get("model")
                    metadata = item.get("meta") or {}
                    context = context or metadata.get("n_ctx")
            except (TypeError, ValueError):
                pass
        if model_id:
            model_id = str(model_id).replace("\\", "/").rsplit("/", 1)[-1]
        model_id = model_id or "local-gguf"
        # Current GGUF packages may advertise a larger load context than
        # their trained context; 64K is the safe advertised ceiling here.
        context = min(int(context or 32768), 65536)
        return model_id, f"{model_id} (this PC)", GGUF_PORT, context

    from .hyperqwen import HYPERQWEN, PORT as HYPERQWEN_PORT  # noqa: PLC0415
    hyperqwen = HYPERQWEN.status()
    if hyperqwen["running"]:
        model_id = "qwen3.8-27b"
        name = hyperqwen.get("model") or "Qwen3.8-27B (this PC)"
        return model_id, name, HYPERQWEN_PORT, 65536

    # Keep the harness installable while no model is running. Starting a
    # model and restarting the harness will regenerate this provider entry.
    model_id = getattr(LLM, "model_id", "qwen3.8-27b") or "qwen3.8-27b"
    context = 65536 if LLM._profile == "c1" else 32768
    return model_id, f"{model_id} (this PC)", LLM_PORT, context


class HarnessManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._logfile = _RUNTIME / "dsh.log"
        self._probe: tuple[float, bool] | None = None
        self._install_lock = threading.Lock()
        self._install_state_lock = threading.Lock()
        self._installing = False
        self._install_stage = ""
        self._install_error = ""

    @property
    def installed(self) -> bool:
        return NODE_EXE.exists() and _dsh_entry_win() is not None

    def healthy(self, timeout: float = 4.0, max_age: float = 0.0) -> bool:
        """The port is the truth: an interop proxy handle can die while
        the Windows process lives on, so never gate this on self._proc.
        Cached (max_age) because each probe spawns a Windows process and
        the dashboard polls constantly — a fresh spawn under load misses
        small timeouts and flaps the UI."""
        now = time.time()
        if max_age and self._probe and now - self._probe[0] < max_age:
            return self._probe[1]
        ok = hostos.http_status(
            f"http://127.0.0.1:{WEB_PORT}/",
            timeout).startswith(("2", "3"))
        self._probe = (now, ok)
        return ok

    @property
    def running(self) -> bool:
        return self.healthy(3, max_age=5)

    def status(self) -> dict:
        alive = self.healthy(3, max_age=5)
        with self._install_state_lock:
            installing = self._installing
            install_stage = self._install_stage
            install_error = self._install_error
        return {
            "installed": self.installed,
            "running": alive,
            "healthy": alive,
            "installing": installing,
            "install_stage": install_stage,
            "install_error": install_error,
            "url_hint": f"http://<this-host>:{WEB_PORT}/",
            "uptime_s": round(time.time() - self._started_at)
            if alive and self._started_at else None,
            "needs_llm": not LLM.running,
        }

    def install(self) -> None:
        """Queue an on-demand install of the portable Node runtime and dsh."""
        with self._install_lock:
            with self._install_state_lock:
                if self._installing:
                    return
                if self.installed:
                    self._install_stage = "DeepSeek Harness is already installed."
                    self._install_error = ""
                    return
                self._installing = True
                self._install_stage = "Preparing the installer…"
                self._install_error = ""
            threading.Thread(target=self._install_worker,
                             name="deepseek-harness-installer",
                             daemon=True).start()

    def _set_install_state(self, stage: str, error: str = "") -> None:
        with self._install_state_lock:
            self._install_stage = stage
            self._install_error = error

    @staticmethod
    def _read_url(url: str, limit: int = 8 * 1024 * 1024) -> bytes:
        request = urllib.request.Request(
            url, headers={"User-Agent": "SiliconNode/DeepSeek-Harness-Installer"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("The installer download was larger than expected.")
        return data

    def _node_release(self, platform_name: str) -> tuple[str, str]:
        index = json.loads(self._read_url(NODE_RELEASE_INDEX).decode("utf-8"))
        asset_name = (f"{platform_name}-zip"
                      if platform_name.startswith("win-") else platform_name)
        for release in index:
            version = release.get("version", "")
            if (re.fullmatch(r"v22\.\d+\.\d+", version)
                    and asset_name in release.get("files", [])):
                archive_ext = "zip" if platform_name.startswith("win-") else "tar.xz"
                filename = f"node-{version}-{platform_name}.{archive_ext}"
                return version, filename
        raise RuntimeError(
            f"No supported Node.js 22 release was found for {platform_name}.")

    def _download_node(self, version: str, filename: str, archive_path: Path) -> None:
        sums_url = f"{NODE_RELEASE_BASE}/{version}/SHASUMS256.txt"
        sums = self._read_url(sums_url, 1024 * 1024).decode("ascii", "replace")
        expected = None
        for line in sums.splitlines():
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line.strip())
            if match and match.group(2) == filename:
                expected = match.group(1).lower()
                break
        if not expected:
            raise RuntimeError("Node.js did not publish a checksum for its archive.")

        url = f"{NODE_RELEASE_BASE}/{version}/{filename}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "SiliconNode/DeepSeek-Harness-Installer"})
        digest = hashlib.sha256()
        received = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            length = int(response.headers.get("Content-Length", "0") or 0)
            if length > MAX_NODE_ARCHIVE_BYTES:
                raise RuntimeError("The Node.js archive exceeded the size limit.")
            with archive_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > MAX_NODE_ARCHIVE_BYTES:
                        raise RuntimeError(
                            "The Node.js archive exceeded the size limit.")
                    output.write(chunk)
                    digest.update(chunk)
                    if length:
                        percent = min(100, round(received * 100 / length))
                        self._set_install_state(
                            f"Downloading Node.js {version}… {percent}%")
                    else:
                        self._set_install_state(
                            f"Downloading Node.js {version}… "
                            f"{received // (1024 * 1024)} MB")
        if digest.hexdigest() != expected:
            raise RuntimeError("Node.js archive checksum verification failed.")

    @staticmethod
    def _npm_command(node_path: Path, npm_cli: Path, args: list[str]) -> list[str]:
        # WSL launches Windows executables through their /mnt/<drive> path;
        # only their file arguments need Windows paths because interop leaves
        # arguments unchanged.
        return [str(node_path), _engine_path(npm_cli), *args]

    def _install_worker(self) -> None:
        staging = None
        backups: list[tuple[Path, Path]] = []
        activated: list[Path] = []
        try:
            _RUNTIME.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".dsh-install-", dir=_RUNTIME))
            machine = platform.machine().lower()
            if machine in ("x86_64", "amd64", "x64"):
                arch = "x64"
            elif machine in ("aarch64", "arm64"):
                arch = "arm64"
            else:
                raise RuntimeError(
                    f"Unsupported CPU architecture: {machine or 'unknown'}.")
            platform_name = f"{'win' if hostos.IS_WSL else 'linux'}-{arch}"
            version, filename = self._node_release(platform_name)
            archive_path = staging / filename
            self._set_install_state(f"Downloading Node.js {version}…")
            self._download_node(version, filename, archive_path)

            self._set_install_state("Verifying and unpacking Node.js…")
            extracted = staging / "unpacked"
            extracted.mkdir()
            if filename.endswith(".zip"):
                with zipfile.ZipFile(archive_path) as archive:
                    _safe_extract_zip(archive, extracted)
            else:
                with tarfile.open(archive_path, "r:xz") as archive:
                    _safe_extract_tar(archive, extracted)
            node_root = extracted / f"node-{version}-{platform_name}"
            node_path = node_root / ("node.exe" if hostos.IS_WSL else "bin/node")
            npm_cli = (node_root / "node_modules/npm/bin/npm-cli.js"
                       if hostos.IS_WSL else
                       node_root / "lib/node_modules/npm/bin/npm-cli.js")
            if not node_path.is_file() or not npm_cli.is_file():
                raise RuntimeError("The Node.js archive is missing Node or npm.")

            staged_node = staging / "node"
            os.replace(node_root, staged_node)
            staged_dsh_root = staging / "dsh"
            staged_dsh_root.mkdir()
            self._set_install_state("Installing the pinned DeepSeek Harness package…")
            npm_deadline = time.monotonic() + 900
            command = self._npm_command(
                staged_node / ("node.exe" if hostos.IS_WSL else "bin/node"),
                staged_node / ("node_modules/npm/bin/npm-cli.js" if hostos.IS_WSL
                               else "lib/node_modules/npm/bin/npm-cli.js"),
                ["install", "--prefix", _engine_path(staged_dsh_root),
                 "--no-save", "--no-audit", "--no-fund",
                 f"--before={PACKAGE_NPM_BEFORE}",
                 "--legacy-peer-deps", PACKAGE_SPEC])
            result = subprocess.run(
                command, cwd=str(staged_dsh_root), capture_output=True,
                text=True, errors="replace",
                timeout=max(1, int(npm_deadline - time.monotonic())))
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-1200:]
                raise RuntimeError(
                    "npm could not install DeepSeek Harness. "
                    + (detail or f"npm exited with code {result.returncode}."))

            for _ in range(5):
                peer_specs = _missing_peer_specs(staged_dsh_root)
                if not peer_specs:
                    break
                self._set_install_state(
                    f"Installing {len(peer_specs)} required harness modules…")
                peer_command = self._npm_command(
                    staged_node / ("node.exe" if hostos.IS_WSL else "bin/node"),
                    staged_node / ("node_modules/npm/bin/npm-cli.js"
                                   if hostos.IS_WSL else
                                   "lib/node_modules/npm/bin/npm-cli.js"),
                    ["install", "--prefix", _engine_path(staged_dsh_root),
                     "--no-save", "--no-audit", "--no-fund",
                     f"--before={PACKAGE_NPM_BEFORE}",
                     "--legacy-peer-deps", PACKAGE_SPEC, *peer_specs])
                remaining_s = max(1, int(npm_deadline - time.monotonic()))
                peer_result = subprocess.run(
                    peer_command, cwd=str(staged_dsh_root), capture_output=True,
                    text=True, errors="replace", timeout=remaining_s)
                if peer_result.returncode:
                    detail = (peer_result.stderr or peer_result.stdout
                              ).strip()[-1200:]
                    raise RuntimeError(
                        "npm could not install DeepSeek Harness peer modules. "
                        + (detail or
                           f"npm exited with code {peer_result.returncode}."))
            else:
                missing_names = ", ".join(
                    spec.rsplit("@", 1)[0]
                    for spec in _missing_peer_specs(staged_dsh_root)[:8])
                raise RuntimeError(
                    "Some DeepSeek Harness peer modules remain unresolved: "
                    + missing_names)

            staged_package = (staged_dsh_root / "node_modules" /
                              "@deepseek-ai" / "dsh")
            if not _dsh_entry(staged_package):
                raise RuntimeError(
                    "npm finished, but the DeepSeek Harness package entry "
                    "point is missing.")

            self._set_install_state("Activating DeepSeek Harness…")
            targets = ((staged_node, _RUNTIME / "node"),
                       (staged_dsh_root, _RUNTIME / "dsh"))
            for staged_path, destination in targets:
                if destination.exists():
                    backup = staging / f"backup-{destination.name}"
                    os.replace(destination, backup)
                    backups.append((backup, destination))
                os.replace(staged_path, destination)
                activated.append(destination)
            for backup, _ in backups:
                shutil.rmtree(backup, ignore_errors=True)
            self._set_install_state("DeepSeek Harness installed. Starting it now…")
        except Exception as exc:  # noqa: BLE001
            log.exception("DeepSeek Harness install failed")
            for destination in reversed(activated):
                if destination.exists():
                    shutil.rmtree(destination, ignore_errors=True)
            for backup, destination in reversed(backups):
                if backup.exists():
                    os.replace(backup, destination)
            message = str(exc).strip() or type(exc).__name__
            self._set_install_state("Installation failed.", message[-1600:])
        finally:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
            with self._install_state_lock:
                self._installing = False

    def start(self, wait_healthy_s: float = 90.0) -> None:
        with self._lock:
            if self.running:
                return
            with self._install_state_lock:
                if self._installing:
                    raise RuntimeError("DeepSeek Harness is still installing.")
            if not self.installed:
                raise RuntimeError(
                    "The harness is not installed yet (portable Node + the "
                    "dsh package under silicon-node/runtime).")
            entry = _dsh_entry_win()
            DSH_HOME_WSL.mkdir(parents=True, exist_ok=True)
            model_id, model_name, model_port, context = \
                _active_model_settings()
            (DSH_HOME_WSL / "settings.yaml").write_text(
                _settings_yaml(model_id, model_name, context, model_port))
            local_only_patch = DSH_HOME_WSL / \
                "silicon-node-local-only.patch.yml"
            local_only_patch.write_text(_local_only_patch_yaml())

            env = dict(os.environ)
            env["DSH_HOME"] = _engine_path(DSH_HOME_WSL)
            env[API_KEY_VAR] = "local"
            if hostos.IS_WSL:
                # Only WSLENV-listed variables cross the interop boundary.
                env["WSLENV"] = (env.get("WSLENV", "").rstrip(":") +
                                 f":DSH_HOME:{API_KEY_VAR}").lstrip(":")
            log.info("starting dsh web on :%d (model %s)…", WEB_PORT,
                     model_id)
            self._kill_instances()
            logfh = open(self._logfile, "ab")  # noqa: SIM115
            self._proc = subprocess.Popen(
                [str(NODE_EXE), entry, "web", "--patch",
                 _engine_path(local_only_patch), "--port", str(WEB_PORT)],
                stdout=logfh, stderr=subprocess.STDOUT,
                cwd=str(DSH_HOME_WSL), env=env)
            logfh.close()
            self._started_at = time.time()
        deadline = time.time() + wait_healthy_s
        while time.time() < deadline:
            if self.healthy():
                log.info("dsh healthy on :%d", WEB_PORT)
                return
            # A dead proxy handle is not proof of failure (the Windows
            # child can outlive it) — only the port decides, at deadline.
            time.sleep(2)
        tail = ""
        try:
            tail = self._logfile.read_text(errors="replace")[-300:]
        except OSError:
            pass
        self.stop()
        raise RuntimeError(
            f"dsh did not come up within {wait_healthy_s:.0f}s. "
            f"Log tail: {tail}")

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                self._proc = None
            self._kill_instances()
            self._started_at = None

    @staticmethod
    def _kill_instances() -> None:
        """Kill only *our* node (the dsh entry in its command line) —
        never every node process, the user runs Node too."""
        try:
            if hostos.IS_WSL:
                script = (
                    "Get-CimInstance Win32_Process -Filter "
                    "\"Name='node.exe'\""
                    " | Where-Object { $_.CommandLine -match 'deepseek-ai' }"
                    " | ForEach-Object "
                    "{ Stop-Process -Id $_.ProcessId -Force }")
                subprocess.run(
                    ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/"
                     "powershell.exe", "-NoProfile", "-Command", script],
                    capture_output=True, timeout=30)
            else:
                subprocess.run(
                    ["pkill", "-f", "deepseek-ai/dsh"],
                    capture_output=True, timeout=20)
        except Exception:  # noqa: BLE001
            log.exception("harness kill failed")


HARNESS = HarnessManager()
