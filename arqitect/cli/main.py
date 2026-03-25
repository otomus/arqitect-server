"""Sentient CLI — start/stop/status/qualify commands.

Replaces start.sh with a Python-native process manager.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from arqitect.config.loader import (
    get_project_root,
    get_whatsapp_dir, get_redis_host_port, get_mcp_port,
    get_bridge_port,
)

# Resolved once at startup by _ensure_venv(); all spawns use this.
PYTHON = sys.executable


def _ensure_venv():
    """Create .venv and install requirements if needed. Sets global PYTHON."""
    global PYTHON
    root = str(get_project_root())
    venv_dir = os.path.join(root, ".venv")
    venv_python = os.path.join(venv_dir, "bin", "python")
    requirements = os.path.join(root, "requirements.txt")

    # Already running inside this venv — nothing to do
    if os.path.realpath(sys.executable).startswith(os.path.realpath(venv_dir)):
        PYTHON = venv_python
        return

    # Create venv if missing
    if not os.path.exists(venv_python):
        print("  Creating virtual environment (.venv)...")
        subprocess.run(
            [sys.executable, "-m", "venv", venv_dir],
            check=True, timeout=60,
        )

    # Install/upgrade requirements if requirements.txt exists
    if os.path.exists(requirements):
        marker = os.path.join(venv_dir, ".deps_installed")
        needs_install = not os.path.exists(marker)
        if not needs_install:
            # Re-install if requirements.txt is newer than marker
            needs_install = (
                os.path.getmtime(requirements) > os.path.getmtime(marker)
            )
        if needs_install:
            print("  Installing dependencies...")
            subprocess.run(
                [venv_python, "-m", "pip", "install", "-q", "-r", requirements],
                check=True, timeout=600,
            )
            # Also install the project itself so arqitect package is importable
            if os.path.exists(os.path.join(root, "setup.py")) or os.path.exists(os.path.join(root, "pyproject.toml")):
                subprocess.run(
                    [venv_python, "-m", "pip", "install", "-q", "-e", root],
                    timeout=120,
                )
            # Touch marker
            with open(marker, "w") as f:
                f.write("")
            print("  Dependencies installed.")

    PYTHON = venv_python

    # Ensure PYTHONPATH includes project root so subprocesses can import arqitect
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")


def _pids_dir() -> str:
    return os.path.join(str(get_project_root()), ".pids")


def _pid_path(name: str) -> str:
    return os.path.join(_pids_dir(), f"{name}.pid")


def _ensure_dirs():
    root = str(get_project_root())
    os.makedirs(_pids_dir(), exist_ok=True)
    os.makedirs(os.path.join(root, "sandbox"), exist_ok=True)
    os.makedirs(os.path.join(root, "nerves"), exist_ok=True)


def _write_pid(name: str, pid: int):
    with open(_pid_path(name), "w") as f:
        f.write(str(pid))


def _read_pid(name: str) -> int | None:
    path = _pid_path(name)
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (ValueError, OSError, FileNotFoundError):
        return None


def _kill_pid(name: str):
    pid = _read_pid(name)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  Stopped {name} (pid {pid})")
        except ProcessLookupError:
            pass
        try:
            os.remove(_pid_path(name))
        except OSError:
            pass


def _is_running(name: str) -> bool:
    pid = _read_pid(name)
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _check_redis():
    try:
        from arqitect.config.loader import get_redis_client
        r = get_redis_client()
        r.ping()
        return True
    except Exception:
        return False


def _start_redis():
    if _check_redis():
        print("  Redis: already running")
        return
    print("  Starting Redis...")
    try:
        proc = subprocess.Popen(
            ["redis-server", "--daemonize", "yes"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=5)
        time.sleep(1)
        if _check_redis():
            print("  Redis: started")
        else:
            print("  Redis: failed to start (is redis-server installed?)")
            sys.exit(1)
    except FileNotFoundError:
        print("  Redis: redis-server not found. Please install Redis.")
        sys.exit(1)


def _start_mcp():
    if _is_running("mcp"):
        print("  MCP Server: already running")
        return
    print("  Starting MCP Server...")
    root = str(get_project_root())
    env = os.environ.copy()
    env["ARQITECT_PROJECT_ROOT"] = root
    proc = subprocess.Popen(
        [PYTHON, "-m", "arqitect.mcp.server"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=root, env=env,
    )
    _write_pid("mcp", proc.pid)
    # Wait for MCP to be ready
    for _ in range(30):
        try:
            import requests
            resp = requests.get(f"http://127.0.0.1:{get_mcp_port()}/health", timeout=2)
            if resp.status_code == 200:
                print("  MCP Server: ready")
                return
        except Exception:
            pass
        time.sleep(1)
    print("  MCP Server: started (waiting for tools to load)")


def _start_brain():
    if _is_running("brain"):
        print("  Brain: already running")
        return
    print("  Starting Brain daemon...")
    root = str(get_project_root())
    log_path = os.path.join(root, "brain.log")
    log_file = open(log_path, "a")
    env = os.environ.copy()
    env["ARQITECT_PROJECT_ROOT"] = root
    proc = subprocess.Popen(
        [PYTHON, "-m", "arqitect.brain.brain", "--daemon"],
        stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
        cwd=root, env=env,
    )
    _write_pid("brain", proc.pid)
    print(f"  Brain: started (pid {proc.pid})")
    print(f"  Brain log: {log_path}")


def _start_bridge():
    if _is_running("bridge"):
        print("  Bridge: already running")
        return
    print("  Starting WebSocket Bridge...")
    root = str(get_project_root())
    env = os.environ.copy()
    env["ARQITECT_PROJECT_ROOT"] = root
    proc = subprocess.Popen(
        [PYTHON, "-m", "arqitect.bridge.server"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=root, env=env,
    )
    _write_pid("bridge", proc.pid)
    port = get_bridge_port()
    print(f"  Bridge: started on port {port} (pid {proc.pid})")


def _start_whatsapp():
    wa_dir = get_whatsapp_dir()
    connector_js = os.path.join(wa_dir, "connector.js")
    if not os.path.exists(connector_js):
        return
    if _is_running("whatsapp"):
        print("  WhatsApp: already running")
        return
    # Check if node_modules exists
    if not os.path.exists(os.path.join(wa_dir, "node_modules")):
        print("  WhatsApp: run 'npm install' in connectors/whatsapp/ first")
        return
    print("  Starting WhatsApp connector...")
    proc = subprocess.Popen(
        ["node", connector_js],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=wa_dir,
    )
    _write_pid("whatsapp", proc.pid)
    print(f"  WhatsApp: started (pid {proc.pid})")


def cmd_init(args):
    """Run interactive setup wizard."""
    from arqitect.cli.wizard import run_wizard
    run_wizard()


def cmd_start(args):
    """Start all Sentient services."""
    print("Starting Arqitect...")
    _ensure_dirs()
    _ensure_venv()
    _start_redis()
    _start_mcp()
    _start_brain()
    _start_bridge()
    _start_whatsapp()
    from arqitect.config.loader import get_bridge_port, get_mcp_url
    print("\nArqitect is online.")
    print(f"  Dashboard: http://localhost:{get_bridge_port()}")
    print(f"  MCP Server: {get_mcp_url()}")


def _kill_stray_brains():
    """Kill any brain processes not tracked in .pids (e.g. from manual runs)."""
    import psutil
    my_pid = _read_pid("brain")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "arqitect.brain.brain" in cmdline and proc.info["pid"] != my_pid:
                proc.terminate()
                print(f"  Killed stray brain (pid {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def cmd_stop(args):
    """Stop all Sentient services."""
    print("Stopping Arqitect...")
    for name in ["whatsapp", "bridge", "brain", "mcp"]:
        _kill_pid(name)
    _kill_stray_brains()
    print("Arqitect stopped.")


def cmd_status(args):
    """Show status of all services."""
    services = ["mcp", "brain", "bridge", "whatsapp"]
    redis_ok = _check_redis()
    print(f"  Redis: {'running' if redis_ok else 'stopped'}")
    for name in services:
        running = _is_running(name)
        pid = _read_pid(name)
        status = f"running (pid {pid})" if running else "stopped"
        print(f"  {name.capitalize()}: {status}")


def cmd_traces(args):
    """Open the trace viewer UI in the browser."""
    from arqitect.cli.traces import serve
    serve(port=args.port)


def cmd_qualify(args):
    """Run brain/nerve qualification."""
    _ensure_venv()
    print("Running qualification...")
    subprocess.run(
        [PYTHON, "-m", "arqitect.critic.qualify_nerve"],
        cwd=str(get_project_root()),
    )


def cmd_adapters(args):
    """Manage LoRA adapters."""
    if args.adapters_command == "list":
        _adapters_list()
    elif args.adapters_command == "pull":
        _adapters_pull(args.name)
    else:
        print("Usage: arqitect adapters {list,pull}")


def _adapters_list():
    """List available adapters from community registry."""
    try:
        from create_arqitect.registry.manifest import get_manifest
        manifest = get_manifest()
        if not manifest:
            print("Could not fetch community registry. Check your network connection.")
            return
        adapters = manifest.get("adapters", [])
        if not adapters:
            print("No adapters available yet.")
            return
        print(f"{'Name':<30} {'Size':<10} {'Description'}")
        print("-" * 70)
        for a in adapters:
            print(f"{a['name']:<30} {a.get('size', '?'):<10} {a.get('description', '')}")
    except ImportError:
        print("Community registry not available. Install create-arqitect package.")


def _adapters_pull(name: str):
    """Download an adapter from community registry."""
    print(f"Pulling adapter '{name}'...")
    try:
        from create_arqitect.registry.manifest import get_manifest
        manifest = get_manifest()
        if not manifest:
            print("Could not fetch registry.")
            return
        adapters = {a["name"]: a for a in manifest.get("adapters", [])}
        if name not in adapters:
            print(f"Adapter '{name}' not found. Run 'arqitect adapters list' to see available.")
            return
        adapter = adapters[name]
        url = adapter.get("url", "")
        if not url:
            print(f"No download URL for adapter '{name}'.")
            return
        import requests
        from arqitect.config.loader import get_models_dir
        dest = os.path.join(get_models_dir(), f"{name}.gguf")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading to {dest}...")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Adapter '{name}' downloaded successfully.")
    except ImportError:
        print("Community registry not available. Install create-arqitect package.")
    except Exception as e:
        print(f"Error: {e}")


def cmd_nerve(args):
    """Manage nerves."""
    if args.nerve_command == "list":
        _nerve_list()
    elif args.nerve_command == "install":
        _nerve_install(args.name)
    else:
        print("Usage: arqitect nerve {list,install}")


def _nerve_list():
    """List installed nerves."""
    from arqitect.config.loader import get_nerves_dir
    nerves_dir = get_nerves_dir()
    if not os.path.exists(nerves_dir):
        print("No nerves directory found.")
        return
    nerves = [d for d in os.listdir(nerves_dir)
              if os.path.isdir(os.path.join(nerves_dir, d))
              and not d.startswith((".", "_", "senses"))]
    if not nerves:
        print("No nerves installed.")
        return
    print(f"{'Name':<40} {'Status'}")
    print("-" * 50)
    for n in sorted(nerves):
        nerve_py = os.path.join(nerves_dir, n, "nerve.py")
        status = "active" if os.path.exists(nerve_py) else "incomplete"
        print(f"{n:<40} {status}")


def _nerve_install(name: str):
    """Install a nerve from community registry."""
    print(f"Installing nerve '{name}'...")
    try:
        from create_arqitect.registry.manifest import get_manifest
        manifest = get_manifest()
        if not manifest:
            print("Could not fetch registry.")
            return
        nerves = {n["name"]: n for n in manifest.get("nerves", [])}
        if name not in nerves:
            print(f"Nerve '{name}' not found in registry.")
            return
        nerve_data = nerves[name]
        from arqitect.config.loader import get_nerves_dir
        nerve_dir = os.path.join(get_nerves_dir(), name)
        os.makedirs(nerve_dir, exist_ok=True)
        # Download nerve.py
        url = nerve_data.get("url", "")
        if url:
            import requests
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
                f.write(resp.text)
            print(f"Nerve '{name}' installed successfully.")
        else:
            print(f"No download URL for nerve '{name}'.")
    except ImportError:
        print("Community registry not available. Install create-arqitect package.")
    except Exception as e:
        print(f"Error: {e}")


def cmd_setup_github(args):
    """Set up GitHub App for this server (standalone, outside full wizard)."""
    from arqitect.config.loader import get_config
    from arqitect.github_app import is_configured, setup_github_app

    if is_configured():
        print("GitHub App is already configured.")
        answer = input("  Reconfigure? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return

    server_name = get_config("name", "Arqitect")
    default_slug = f"arqitect-{server_name.lower().replace(' ', '-')}"
    slug = input(f"  GitHub App name [{default_slug}]: ").strip() or default_slug

    setup_github_app(slug)


def cmd_contribute(args):
    """Package a nerve and submit a PR to otomus/arqitect-community."""
    name = args.name
    root = str(get_project_root())
    community_repo = "otomus/arqitect-community"
    community_dir = os.path.join(root, ".community", "arqitect-community")

    # Validate nerve exists
    from arqitect.config.loader import get_nerves_dir
    nerve_dir = os.path.join(get_nerves_dir(), name)
    if not os.path.isdir(nerve_dir):
        print(f"Error: nerve '{name}' not found at {nerve_dir}", file=sys.stderr)
        sys.exit(1)
    nerve_py = os.path.join(nerve_dir, "nerve.py")
    if not os.path.exists(nerve_py):
        print(f"Error: {nerve_py} not found", file=sys.stderr)
        sys.exit(1)

    # Check gh CLI
    try:
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("Error: gh CLI not authenticated. Run 'gh auth login' first.", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("Error: gh CLI not found. Install from https://cli.github.com/", file=sys.stderr)
        sys.exit(1)

    # Fork/clone community repo
    os.makedirs(os.path.dirname(community_dir), exist_ok=True)
    if not os.path.isdir(community_dir):
        print("Forking and cloning community repo...")
        subprocess.run(
            ["gh", "repo", "fork", community_repo, "--clone", "--remote"],
            cwd=os.path.dirname(community_dir), check=True, timeout=60,
        )
    else:
        subprocess.run(["git", "checkout", "main"], cwd=community_dir, check=True, timeout=10)
        subprocess.run(["git", "pull", "origin", "main"], cwd=community_dir, check=True, timeout=30)

    # Copy nerve to community repo
    import shutil
    dest = os.path.join(community_dir, "nerves", name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(nerve_dir, dest)

    # Create branch, commit, push
    branch = f"contribute/{name}"
    subprocess.run(["git", "checkout", "-b", branch], cwd=community_dir, check=True, timeout=10)
    subprocess.run(["git", "add", f"nerves/{name}"], cwd=community_dir, check=True, timeout=10)

    commit_msg = f"Add nerve: {name}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=community_dir, check=True, timeout=10)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=community_dir, check=True, timeout=30)

    # Create PR
    result = subprocess.run(
        ["gh", "pr", "create",
         "--repo", community_repo,
         "--title", f"Add nerve: {name}",
         "--body", f"## Nerve: {name}\n\nContributed from an arqitect-server project."],
        cwd=community_dir, capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        print(f"PR created: {result.stdout.strip()}")
    else:
        print(f"PR error: {result.stderr}", file=sys.stderr)
        print("Your changes are pushed. You can create the PR manually.")


def main():
    parser = argparse.ArgumentParser(
        prog="arqitect",
        description="Sentient — autonomous AI nervous system",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Run interactive setup wizard")
    sub.add_parser("start", help="Start all services")
    sub.add_parser("stop", help="Stop all services")
    sub.add_parser("status", help="Show service status")
    sub.add_parser("qualify", help="Run qualification")
    traces_parser = sub.add_parser("traces", help="Open trace viewer UI")
    traces_parser.add_argument("--port", type=int, default=7681, help="Port (default: 7681)")

    # Adapters
    adapters_parser = sub.add_parser("adapters", help="Manage LoRA adapters")
    adapters_sub = adapters_parser.add_subparsers(dest="adapters_command")
    adapters_sub.add_parser("list", help="List available adapters")
    adapters_pull = adapters_sub.add_parser("pull", help="Pull an adapter")
    adapters_pull.add_argument("name", help="Adapter name")

    # Nerves
    nerve_parser = sub.add_parser("nerve", help="Manage nerves")
    nerve_sub = nerve_parser.add_subparsers(dest="nerve_command")
    nerve_sub.add_parser("list", help="List installed nerves")
    nerve_install = nerve_sub.add_parser("install", help="Install a nerve")
    nerve_install.add_argument("name", help="Nerve name")

    # GitHub App setup
    sub.add_parser("setup-github", help="Set up GitHub App for this server")

    # Contribute
    contribute_parser = sub.add_parser("contribute", help="Package a nerve for community")
    contribute_parser.add_argument("name", help="Nerve name to contribute")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "qualify": cmd_qualify,
        "traces": cmd_traces,
        "adapters": cmd_adapters,
        "nerve": cmd_nerve,
        "setup-github": cmd_setup_github,
        "contribute": cmd_contribute,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
