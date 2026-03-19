"""Sense: Touch — file CRUD, OS actions, system manipulation.

Commands: read, write, list, exists, delete, mkdir, info, exec, sysinfo
Calls awareness sense before destructive operations.
Sandboxed: operates within sandbox/ by default, refuses system paths.
"""

import json
import os
import platform
import shutil
import subprocess
import sys

_SENSE_DIR = os.path.dirname(os.path.abspath(__file__))
from arqitect.config.loader import get_project_root, get_sandbox_dir as _get_sandbox_dir
_PROJECT_ROOT = str(get_project_root())
_SANDBOX_DIR = _get_sandbox_dir()

SENSE_NAME = "touch"
def _load_adapter_description() -> str:
    try:
        from arqitect.brain.adapters import get_description
        desc = get_description("touch")
        if desc:
            return desc
    except Exception:
        pass
    return "File CRUD, OS actions, and system manipulation"

DESCRIPTION = _load_adapter_description()

# Paths that are always denied (system-critical)
_SYSTEM_PATHS = frozenset({
    "/etc", "/usr", "/bin", "/sbin", "/var", "/System",
    "/Library", "/boot", "/dev", "/proc", "/sys",
})


def _resolve_path(path: str) -> str:
    """Resolve a path, defaulting to sandbox directory."""
    if not path:
        return _SANDBOX_DIR
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_SANDBOX_DIR, path)
    return os.path.normpath(path)


def _is_system_path(path: str) -> bool:
    """Check if a path is in a system-critical directory."""
    path = os.path.normpath(path)
    for sp in _SYSTEM_PATHS:
        if path == sp or path.startswith(sp + os.sep):
            return True
    return False


def _check_awareness(action: str, context: str) -> dict:
    """Check with awareness sense before destructive operations."""
    try:
        from arqitect.senses.sense_runtime import check_awareness
        return check_awareness(action, context)
    except Exception:
        # If awareness unavailable, allow but warn
        return {"allowed": True, "warning": "Awareness sense unavailable"}


def cmd_read(path: str) -> dict:
    """Read file contents."""
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        return {"error": f"File not found: {resolved}"}
    if os.path.isdir(resolved):
        return {"error": f"Path is a directory, use 'list' command: {resolved}"}
    try:
        with open(resolved, "r") as f:
            content = f.read()
        return {"path": resolved, "content": content, "size": len(content)}
    except Exception as e:
        return {"error": f"Cannot read {resolved}: {e}"}


def cmd_write(path: str, content: str) -> dict:
    """Write content to a file."""
    resolved = _resolve_path(path)
    if _is_system_path(resolved):
        return {"error": f"Refused: cannot write to system path {resolved}"}
    # Check awareness for writes to non-sandbox paths
    if not resolved.startswith(os.path.normpath(_SANDBOX_DIR)):
        check = _check_awareness("write", f"path={resolved}")
        if check.get("denied"):
            return {"error": check.get("reason", "Write denied by awareness")}
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    try:
        with open(resolved, "w") as f:
            f.write(content)
        return {"path": resolved, "written": len(content), "success": True}
    except Exception as e:
        return {"error": f"Cannot write {resolved}: {e}"}


def cmd_list(path: str) -> dict:
    """List directory contents."""
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        return {"error": f"Path not found: {resolved}"}
    if not os.path.isdir(resolved):
        return {"error": f"Not a directory: {resolved}"}
    try:
        entries = []
        for name in sorted(os.listdir(resolved)):
            full = os.path.join(resolved, name)
            entry = {"name": name, "type": "dir" if os.path.isdir(full) else "file"}
            if os.path.isfile(full):
                entry["size"] = os.path.getsize(full)
            entries.append(entry)
        return {"path": resolved, "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": f"Cannot list {resolved}: {e}"}


def cmd_exists(path: str) -> dict:
    """Check if a path exists."""
    resolved = _resolve_path(path)
    exists = os.path.exists(resolved)
    result = {"path": resolved, "exists": exists}
    if exists:
        result["type"] = "dir" if os.path.isdir(resolved) else "file"
        if os.path.isfile(resolved):
            result["size"] = os.path.getsize(resolved)
    return result


def cmd_delete(path: str) -> dict:
    """Delete a file or directory (with awareness check)."""
    resolved = _resolve_path(path)
    if _is_system_path(resolved):
        return {"error": f"Refused: cannot delete system path {resolved}"}
    if not os.path.exists(resolved):
        return {"error": f"Path not found: {resolved}"}
    # Always check awareness for deletes
    check = _check_awareness("delete", f"path={resolved}")
    if check.get("denied"):
        return {"error": check.get("reason", "Delete denied by awareness")}
    try:
        if os.path.isdir(resolved):
            shutil.rmtree(resolved)
        else:
            os.remove(resolved)
        return {"path": resolved, "deleted": True}
    except Exception as e:
        return {"error": f"Cannot delete {resolved}: {e}"}


def cmd_mkdir(path: str) -> dict:
    """Create a directory."""
    resolved = _resolve_path(path)
    if _is_system_path(resolved):
        return {"error": f"Refused: cannot create directory in system path {resolved}"}
    try:
        os.makedirs(resolved, exist_ok=True)
        return {"path": resolved, "created": True}
    except Exception as e:
        return {"error": f"Cannot create {resolved}: {e}"}


def cmd_info(path: str) -> dict:
    """Get file/directory metadata."""
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        return {"error": f"Path not found: {resolved}"}
    try:
        stat = os.stat(resolved)
        return {
            "path": resolved,
            "type": "dir" if os.path.isdir(resolved) else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:],
        }
    except Exception as e:
        return {"error": f"Cannot get info for {resolved}: {e}"}


def cmd_append(path: str, content: str) -> dict:
    """Append content to a file."""
    resolved = _resolve_path(path)
    if _is_system_path(resolved):
        return {"error": f"Refused: cannot write to system path {resolved}"}
    if not resolved.startswith(os.path.normpath(_SANDBOX_DIR)):
        check = _check_awareness("append", f"path={resolved}")
        if check.get("denied"):
            return {"error": check.get("reason", "Append denied by awareness")}
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    try:
        with open(resolved, "a") as f:
            f.write(content)
        return {"path": resolved, "appended": len(content), "success": True}
    except Exception as e:
        return {"error": f"Cannot append to {resolved}: {e}"}


def cmd_copy(src: str, dest: str) -> dict:
    """Copy a file or directory."""
    resolved_src = _resolve_path(src)
    resolved_dest = _resolve_path(dest)
    if not os.path.exists(resolved_src):
        return {"error": f"Source not found: {resolved_src}"}
    if _is_system_path(resolved_dest):
        return {"error": f"Refused: cannot copy to system path {resolved_dest}"}
    try:
        if os.path.isdir(resolved_src):
            shutil.copytree(resolved_src, resolved_dest, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(resolved_dest), exist_ok=True)
            shutil.copy2(resolved_src, resolved_dest)
        return {"src": resolved_src, "dest": resolved_dest, "copied": True}
    except Exception as e:
        return {"error": f"Cannot copy: {e}"}


def cmd_move(src: str, dest: str) -> dict:
    """Move or rename a file or directory."""
    resolved_src = _resolve_path(src)
    resolved_dest = _resolve_path(dest)
    if not os.path.exists(resolved_src):
        return {"error": f"Source not found: {resolved_src}"}
    if _is_system_path(resolved_dest):
        return {"error": f"Refused: cannot move to system path {resolved_dest}"}
    try:
        os.makedirs(os.path.dirname(resolved_dest), exist_ok=True)
        shutil.move(resolved_src, resolved_dest)
        return {"src": resolved_src, "dest": resolved_dest, "moved": True}
    except Exception as e:
        return {"error": f"Cannot move: {e}"}


def cmd_search(path: str, pattern: str) -> dict:
    """Search for files matching a glob pattern recursively."""
    import fnmatch
    resolved = _resolve_path(path) if path else _SANDBOX_DIR
    if not os.path.isdir(resolved):
        return {"error": f"Directory not found: {resolved}"}
    matches = []
    for root, dirs, files in os.walk(resolved):
        for name in files + dirs:
            if fnmatch.fnmatch(name, pattern):
                full = os.path.join(root, name)
                matches.append({"path": full, "type": "dir" if os.path.isdir(full) else "file"})
                if len(matches) >= 100:
                    break
        if len(matches) >= 100:
            break
    return {"pattern": pattern, "root": resolved, "matches": matches, "count": len(matches)}


def cmd_tree(path: str, max_depth: int = 3) -> dict:
    """Recursive directory listing as a tree."""
    resolved = _resolve_path(path) if path else _SANDBOX_DIR
    if not os.path.isdir(resolved):
        return {"error": f"Directory not found: {resolved}"}

    def _walk(dir_path, depth):
        if depth > max_depth:
            return []
        entries = []
        try:
            items = sorted(os.listdir(dir_path))
        except PermissionError:
            return [{"name": "(permission denied)", "type": "error"}]
        for name in items:
            if name.startswith("."):
                continue
            full = os.path.join(dir_path, name)
            if os.path.isdir(full):
                children = _walk(full, depth + 1)
                entries.append({"name": name, "type": "dir", "children": children})
            else:
                entries.append({"name": name, "type": "file", "size": os.path.getsize(full)})
        return entries

    tree = _walk(resolved, 1)
    return {"path": resolved, "tree": tree}


def cmd_exec(command: str) -> dict:
    """Execute a shell command (with awareness check)."""
    # Always check awareness for exec
    check = _check_awareness("exec", f"command={command}")
    if check.get("denied"):
        return {"error": check.get("reason", "Exec denied by awareness")}
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_SANDBOX_DIR,
        )
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out: {command}"}
    except Exception as e:
        return {"error": f"Exec failed: {e}"}


def cmd_sysinfo() -> dict:
    """Get system information."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "sandbox": _SANDBOX_DIR,
    }


_COMMANDS = {
    "read": lambda d: cmd_read(d.get("path", "")),
    "write": lambda d: cmd_write(d.get("path", ""), d.get("content", "")),
    "append": lambda d: cmd_append(d.get("path", ""), d.get("content", "")),
    "list": lambda d: cmd_list(d.get("path", "")),
    "ls": lambda d: cmd_list(d.get("path", "")),
    "tree": lambda d: cmd_tree(d.get("path", ""), d.get("max_depth", 3)),
    "search": lambda d: cmd_search(d.get("path", ""), d.get("pattern", "*")),
    "find": lambda d: cmd_search(d.get("path", ""), d.get("pattern", "*")),
    "exists": lambda d: cmd_exists(d.get("path", "")),
    "copy": lambda d: cmd_copy(d.get("src", d.get("path", "")), d.get("dest", "")),
    "cp": lambda d: cmd_copy(d.get("src", d.get("path", "")), d.get("dest", "")),
    "move": lambda d: cmd_move(d.get("src", d.get("path", "")), d.get("dest", "")),
    "mv": lambda d: cmd_move(d.get("src", d.get("path", "")), d.get("dest", "")),
    "rename": lambda d: cmd_move(d.get("src", d.get("path", "")), d.get("dest", "")),
    "delete": lambda d: cmd_delete(d.get("path", "")),
    "rm": lambda d: cmd_delete(d.get("path", "")),
    "mkdir": lambda d: cmd_mkdir(d.get("path", "")),
    "info": lambda d: cmd_info(d.get("path", "")),
    "exec": lambda d: cmd_exec(d.get("cmd", d.get("command", ""))),
    "sysinfo": lambda d: cmd_sysinfo(),
}


def calibrate() -> dict:
    """Probe touch sense capabilities."""
    from arqitect.senses.calibration_protocol import build_result, save_calibration

    # Touch always works — it's pure Python + filesystem
    sandbox_writable = os.access(_SANDBOX_DIR, os.W_OK) if os.path.exists(_SANDBOX_DIR) else True
    exec_available = True  # subprocess always available

    capabilities = {
        "read": {"available": True, "provider": "filesystem", "notes": ""},
        "write": {"available": sandbox_writable, "provider": "filesystem",
                  "notes": "" if sandbox_writable else "Sandbox directory not writable"},
        "append": {"available": sandbox_writable, "provider": "filesystem", "notes": ""},
        "list": {"available": True, "provider": "filesystem", "notes": ""},
        "tree": {"available": True, "provider": "filesystem", "notes": "Recursive directory listing"},
        "search": {"available": True, "provider": "filesystem", "notes": "Glob pattern search"},
        "copy": {"available": True, "provider": "filesystem", "notes": ""},
        "move": {"available": True, "provider": "filesystem", "notes": ""},
        "delete": {"available": True, "provider": "filesystem", "notes": "Awareness-gated"},
        "exec": {"available": exec_available, "provider": "subprocess", "notes": "Awareness-gated"},
        "sysinfo": {"available": True, "provider": "platform", "notes": ""},
    }

    deps = {
        "sandbox_dir": {"installed": os.path.isdir(_SANDBOX_DIR), "path": _SANDBOX_DIR},
    }

    result = build_result(SENSE_NAME, capabilities, deps)
    save_calibration(_SENSE_DIR, result)
    return result


def _parse_natural_language(raw: str) -> dict:
    """Parse natural language input into a structured command.

    Handles inputs like:
      'list /some/path'
      'read /some/path/file.py'
      'write /some/path/file.py content here'
      'exec ls -la /tmp'
      'create /some/dir'
    """
    raw = raw.strip()

    # Try to detect command from first word
    parts = raw.split(None, 1)
    if not parts:
        return {"command": "list", "path": ""}

    first = parts[0].lower().rstrip(",:")
    rest = parts[1] if len(parts) > 1 else ""

    # Map common words to commands
    cmd_map = {
        "read": "read", "cat": "read", "show": "read", "open": "read", "view": "read",
        "write": "write", "save": "write", "create": "write",
        "append": "append",
        "list": "list", "ls": "list", "dir": "list",
        "tree": "tree",
        "search": "search", "find": "search", "grep": "search",
        "copy": "copy", "cp": "copy",
        "move": "move", "mv": "move", "rename": "move",
        "delete": "delete", "rm": "delete", "remove": "delete",
        "mkdir": "mkdir",
        "exec": "exec", "run": "exec", "execute": "exec",
        "sysinfo": "sysinfo", "info": "info",
        "exists": "exists", "check": "exists",
    }

    command = cmd_map.get(first)
    if command:
        if command in ("write", "append"):
            # Split rest into path and content
            rest_parts = rest.split(None, 1)
            if rest_parts:
                path = rest_parts[0]
                content = rest_parts[1] if len(rest_parts) > 1 else ""
                for prefix in ("content:", "with content:", "with content ", "containing "):
                    if content.lower().startswith(prefix):
                        content = content[len(prefix):].strip()
                        break
                return {"command": command, "path": path, "content": content}
            return {"command": command, "path": "", "content": ""}
        elif command in ("copy", "move"):
            # Expect: copy/move <src> <dest> or <src> to <dest>
            rest_clean = rest.replace(" to ", " ").strip()
            parts2 = rest_clean.split(None, 1)
            src = parts2[0] if parts2 else ""
            dest = parts2[1].strip() if len(parts2) > 1 else ""
            return {"command": command, "src": src, "dest": dest}
        elif command == "exec":
            return {"command": "exec", "cmd": rest}
        elif command == "sysinfo":
            return {"command": "sysinfo"}
        elif command == "search":
            # search <pattern> in <path> or search <path> <pattern>
            if " in " in rest:
                pattern, path = rest.split(" in ", 1)
                return {"command": "search", "pattern": pattern.strip(), "path": path.strip()}
            rest_parts = rest.split(None, 1)
            if len(rest_parts) == 2:
                return {"command": "search", "path": rest_parts[0], "pattern": rest_parts[1]}
            return {"command": "search", "path": "", "pattern": rest.strip()}
        else:
            return {"command": command, "path": rest.strip()}

    # No recognized command — if input looks like a path, default to read/list
    if "/" in raw or raw.startswith("~"):
        path = raw.split()[0]
        if os.path.splitext(path)[1]:  # has extension → read
            return {"command": "read", "path": path}
        return {"command": "list", "path": path}

    return {"command": "list", "path": raw}


def main():
    # Args come as a single argument from invoke_nerve, or multiple from CLI
    if len(sys.argv) == 2:
        raw = sys.argv[1]
    elif len(sys.argv) > 2:
        raw = " ".join(sys.argv[1:])
    else:
        raw = "{}"
    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError:
        # Parse as natural language
        input_data = _parse_natural_language(raw)

    # Calibration mode
    if input_data.get("mode") == "calibrate":
        print(json.dumps(calibrate()))
        return

    command = input_data.get("command", "list").lower().strip()
    handler = _COMMANDS.get(command)
    if not handler:
        result = {"error": f"Unknown command: {command}", "available": list(_COMMANDS.keys())}
    else:
        os.makedirs(_SANDBOX_DIR, exist_ok=True)
        result = handler(input_data)

    result["sense"] = SENSE_NAME
    print(json.dumps(result))


if __name__ == "__main__":
    main()
