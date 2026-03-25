"""GitHub App client for authenticated contributions.

Each arqitect server registers its own GitHub App during setup. The app's
private key is stored in ``arqitect.yaml`` secrets and used to mint
short-lived installation tokens scoped to specific repositories.

The app identity (``arqitect-{name}[bot]``) is traceable per-server.
Access to any repository requires the repo owner to install the app first.

Typical flow:
1. ``setup_github_app()`` — called once during ``arqitect init``
2. ``get_installation_token(repo)`` — called per-contribution to mint a token
3. ``github_api()`` / ``create_pr()`` — authenticated GitHub API calls
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

import jwt
from pathlib import Path

import requests

from arqitect.config.loader import get_config, get_project_root, get_secret, set_secret

logger = logging.getLogger(__name__)

COMMUNITY_REPO = "otomus/arqitect-community"
SECRETS_DIR = ".secrets"
PRIVATE_KEY_FILENAME = "github_app.pem"


def _load_private_key() -> str:
    """Load the GitHub App private key from disk.

    Reads the key path from config (``github.private_key_path``) and
    returns the PEM content. Falls back to the legacy inline
    ``github.private_key`` secret for backward compatibility.

    Returns:
        PEM string, or empty string if not configured.
    """
    key_path = get_secret("github.private_key_path", "")
    if key_path:
        try:
            return Path(key_path).read_text()
        except (FileNotFoundError, PermissionError) as exc:
            logger.warning("[GITHUB] Cannot read private key at %s: %s", key_path, exc)
            return ""

    # Backward compat: key stored inline in arqitect.yaml
    return get_secret("github.private_key", "")


# ── JWT Generation ────────────────────────────────────────────────────────

def _build_jwt(app_id: str, private_key: str) -> str:
    """Build a short-lived JWT for GitHub App authentication.

    Args:
        app_id: GitHub App ID.
        private_key: PEM-encoded RSA private key.

    Returns:
        Signed JWT string valid for 10 minutes.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (10 * 60),
        "iss": app_id,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


# ── Installation Token ────────────────────────────────────────────────────

_token_cache: dict[str, tuple[str, float]] = {}


def get_installation_token(repo: str = COMMUNITY_REPO) -> str | None:
    """Get a short-lived installation token scoped to a repository.

    Caches tokens for 50 minutes (they expire after 60).

    Args:
        repo: Repository in ``owner/name`` format.

    Returns:
        Installation token string, or None if the app is not configured
        or not installed on the target repo.
    """
    now = time.time()
    cached = _token_cache.get(repo)
    if cached and cached[1] > now:
        return cached[0]

    app_id = get_secret("github.app_id", "")
    private_key = _load_private_key()
    if not app_id or not private_key:
        logger.debug("[GITHUB] App not configured (no app_id or private_key)")
        return None

    try:
        app_jwt = _build_jwt(app_id, private_key)

        # Find installation for the target repo
        installation_id = _find_installation(app_jwt, repo)
        if not installation_id:
            logger.warning("[GITHUB] App not installed on '%s'", repo)
            return None

        # Mint installation token
        resp = requests.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        _token_cache[repo] = (token, now + 50 * 60)
        return token

    except Exception as e:
        logger.warning("[GITHUB] Failed to get installation token for '%s': %s", repo, e)
        return None


def _find_installation(app_jwt: str, repo: str) -> int | None:
    """Find the installation ID for a specific repository.

    Args:
        app_jwt: JWT signed with the app's private key.
        repo: Repository in ``owner/name`` format.

    Returns:
        Installation ID, or None if not found.
    """
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/installation",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()["id"]
        return None
    except Exception:
        return None


# ── Authenticated API Calls ───────────────────────────────────────────────

def github_api(method: str, url: str, repo: str = COMMUNITY_REPO,
               body: dict | None = None, timeout: int = 15) -> dict | None:
    """Make an authenticated GitHub API call using the app's installation token.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE).
        url: Full GitHub API URL.
        repo: Repository for token scoping.
        body: Optional JSON body for POST/PUT/PATCH.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response, or None on failure.
    """
    token = get_installation_token(repo)
    if not token:
        return None

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        resp = requests.request(method, url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
    except Exception as e:
        logger.warning("[GITHUB] API %s %s failed: %s", method, url, e)
        return None


def create_pr(repo: str, title: str, body: str, head: str,
              base: str = "main") -> str | None:
    """Create a pull request on a repository.

    Args:
        repo: Repository in ``owner/name`` format.
        title: PR title.
        body: PR description (markdown).
        head: Source branch name.
        base: Target branch name.

    Returns:
        PR URL on success, None on failure.
    """
    result = github_api(
        "POST",
        f"https://api.github.com/repos/{repo}/pulls",
        repo=repo,
        body={"title": title, "body": body, "head": head, "base": base},
    )
    if result and "html_url" in result:
        return result["html_url"]
    return None


def list_open_prs(repo: str, author: str | None = None) -> list[dict]:
    """List open PRs on a repository, optionally filtered by author.

    Args:
        repo: Repository in ``owner/name`` format.
        author: GitHub username to filter by (None for all).

    Returns:
        List of PR dicts with number, title, headRefName, etc.
    """
    token = get_installation_token(repo)
    if not token:
        return []

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        params = {"state": "open", "per_page": 50}
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/pulls",
            headers=headers, params=params, timeout=15,
        )
        resp.raise_for_status()
        prs = resp.json()
        if author:
            prs = [p for p in prs if p.get("user", {}).get("login", "") == author]
        return prs
    except Exception as e:
        logger.warning("[GITHUB] Failed to list PRs on '%s': %s", repo, e)
        return []


def get_app_slug() -> str:
    """Get the app's slug (login name) for filtering PRs by author.

    Returns:
        App slug like ``arqitect-ob1[bot]``, or empty string.
    """
    app_id = get_secret("github.app_id", "")
    private_key = _load_private_key()
    if not app_id or not private_key:
        return ""
    try:
        app_jwt = _build_jwt(app_id, private_key)
        resp = requests.get(
            "https://api.github.com/app",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        slug = resp.json().get("slug", "")
        return f"{slug}[bot]" if slug else ""
    except Exception:
        return ""


# ── Git Authentication Helper ─────────────────────────────────────────────

def configure_git_auth(repo_dir: str, repo: str = COMMUNITY_REPO) -> bool:
    """Configure git credentials for a repository directory using the app token.

    Sets the remote URL to use the installation token for push/pull.
    Also configures git user identity from the app.

    Args:
        repo_dir: Path to the git repository.
        repo: Repository in ``owner/name`` format.

    Returns:
        True if configured successfully, False otherwise.
    """
    token = get_installation_token(repo)
    if not token:
        return False

    app_name = get_config("name", "Arqitect")
    app_id = get_secret("github.app_id", "")

    try:
        _run_git(repo_dir, ["git", "remote", "set-url", "origin",
                             f"https://x-access-token:{token}@github.com/{repo}.git"])
        _run_git(repo_dir, ["git", "config", "user.name", f"{app_name}[bot]"])
        _run_git(repo_dir, ["git", "config", "user.email",
                             f"{app_id}+{app_name.lower()}[bot]@users.noreply.github.com"])
        return True
    except Exception as e:
        logger.warning("[GITHUB] Failed to configure git auth: %s", e)
        return False


def _run_git(cwd: str, cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    """Run a git command in a directory.

    Args:
        cwd: Working directory.
        cmd: Command and arguments.
        timeout: Timeout in seconds.

    Returns:
        Completed process result.
    """
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# ── Setup ─────────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Check if the GitHub App is configured with credentials."""
    return bool(get_secret("github.app_id", "")) and bool(_load_private_key())


def _build_manifest(slug: str, redirect_url: str) -> dict:
    """Build the GitHub App manifest payload.

    Args:
        slug: Lowercase app name (e.g. ``arqitect-ob1``).
        redirect_url: Local callback URL for the manifest flow.

    Returns:
        Manifest dict ready for JSON serialization.
    """
    return {
        "name": slug,
        "url": "https://github.com/otomus/arqitect",
        "hook_attributes": {"active": False},
        "redirect_url": redirect_url,
        "public": False,
        "default_permissions": {
            "contents": "write",
            "pull_requests": "write",
            "metadata": "read",
        },
        "default_events": [],
    }


def _exchange_manifest_code(code: str) -> dict | None:
    """Exchange a manifest flow code for app credentials.

    Args:
        code: One-time code from the GitHub redirect.

    Returns:
        App data dict with ``id`` and ``pem``, or None on failure.
    """
    try:
        resp = requests.post(
            f"https://api.github.com/app-manifests/{code}/conversions",
            headers={"Accept": "application/vnd.github+json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("[GITHUB] Manifest code exchange failed: %s", exc)
        return None


def _save_app_credentials(app_data: dict, slug: str) -> None:
    """Persist app credentials to arqitect.yaml secrets.

    Args:
        app_data: Response from the manifest code exchange.
        slug: App slug for display.
    """
    key_path = _store_private_key(app_data["pem"])
    set_secret("github.app_id", str(app_data["id"]))
    set_secret("github.private_key_path", str(key_path))
    set_secret("github.app_slug", slug)


def _setup_via_manifest(slug: str) -> bool:
    """Create a GitHub App using the manifest flow (browser-based).

    Opens the browser directly to GitHub's manifest creation URL with the
    manifest as a query parameter. After the user approves, GitHub
    redirects back to a local callback server with a one-time code, which
    is exchanged for app credentials.

    Args:
        slug: Lowercase app name.

    Returns:
        True if the app was created and credentials stored.
    """
    code_holder: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            params = parse_qs(urlparse(self.path).query)
            received = params.get("code", [""])[0]

            if received:
                code_holder.append(received)
                self._respond(
                    "<h2>GitHub App created!</h2>"
                    "<p>You can close this tab and return to the terminal.</p>"
                )
            else:
                self.send_response(204)
                self.end_headers()

        def _respond(self, body: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    server.timeout = 10

    redirect_url = f"http://localhost:{port}"
    manifest = _build_manifest(slug, redirect_url)
    manifest_encoded = quote(json.dumps(manifest, separators=(",", ":")))
    github_url = f"https://github.com/settings/apps/new?manifest={manifest_encoded}"

    print(f"  Opening browser to create GitHub App '{slug}'...")
    print(f"  If the browser doesn't open, visit:\n  {github_url}\n")
    webbrowser.open(github_url)
    print("  Waiting for authorization (up to 2 minutes)...")

    deadline = time.time() + 120
    while not code_holder and time.time() < deadline:
        try:
            server.handle_request()
        except Exception:
            break
    server.server_close()

    if not code_holder:
        print("  Timed out waiting for GitHub redirect.")
        return False

    # Exchange the one-time code for app credentials
    print("  Exchanging code for app credentials...")
    app_data = _exchange_manifest_code(code_holder[0])
    if not app_data or "id" not in app_data or "pem" not in app_data:
        print("  Failed to exchange code for credentials.")
        return False

    _save_app_credentials(app_data, slug)

    print(f"\n  GitHub App '{slug}' created (ID: {app_data['id']})")
    print(f"  Now install it on {COMMUNITY_REPO}:")
    print(f"  https://github.com/settings/apps/{slug}/installations/new")
    return True


def _get_private_key_path() -> Path:
    """Return the canonical path for the GitHub App private key.

    Returns:
        ``{project_root}/.secrets/github_app.pem``
    """
    return get_project_root() / SECRETS_DIR / PRIVATE_KEY_FILENAME


def _store_private_key(pem: str) -> Path:
    """Save the private key PEM to the project's secrets directory.

    Creates ``{project_root}/.secrets/`` with restricted permissions
    (owner-only) and writes ``github_app.pem`` inside it.

    Args:
        pem: PEM-encoded RSA private key string.

    Returns:
        Path to the saved key file.
    """
    key_path = _get_private_key_path()
    key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_path.write_text(pem)
    key_path.chmod(0o600)
    logger.info("[GITHUB] Private key saved to %s", key_path)
    return key_path


def _open_pem_file_dialog() -> str:
    """Open a native file picker for .pem files.

    Returns:
        Selected file path, or empty string if cancelled/unavailable.
    """
    import platform
    system = platform.system()

    try:
        if system == "Darwin":
            result = subprocess.run(
                ["osascript", "-e",
                 'set theFile to choose file with prompt "Select .pem private key file"',
                 "-e", "return POSIX path of theFile"],
                capture_output=True, text=True, timeout=120,
            )
            return result.stdout.strip() if result.returncode == 0 else ""

        if system == "Windows":
            ps_script = (
                "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null;"
                "$dlg = New-Object System.Windows.Forms.OpenFileDialog;"
                '$dlg.Title = "Select .pem private key file";'
                '$dlg.Filter = "PEM files (*.pem)|*.pem|All files (*.*)|*.*";'
                "if ($dlg.ShowDialog() -eq 'OK') { $dlg.FileName } else { '' }"
            )
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True, text=True, timeout=120,
            )
            return result.stdout.strip() if result.returncode == 0 else ""

        # Linux: zenity
        result = subprocess.run(
            ["zenity", "--file-selection", "--title=Select .pem private key file",
             "--file-filter", "PEM files | *.pem"],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _read_private_key() -> str:
    """Read a private key via file picker, file path, or pasted PEM text.

    Opens a native file picker first. If cancelled, falls back to
    manual input (file path or pasted PEM).

    Returns:
        PEM-encoded private key string.
    """
    print("  Opening file picker for .pem key...")
    path = _open_pem_file_dialog()
    if path:
        print(f"  Selected: {path}")
        with open(path) as f:
            return f.read()

    print("  No file selected.")
    choice = input("  Private key — enter .pem file path or paste PEM: ").strip()

    # If it looks like a file path, read it
    if choice and not choice.startswith("-----"):
        try:
            with open(choice) as f:
                return f.read()
        except FileNotFoundError:
            print(f"  File not found: {choice}")
            print("  Paste the PEM key instead (press Enter twice when done):")

    # Collect multiline PEM input
    lines = [choice] if choice.startswith("-----") else []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines) + "\n"


def _setup_manually(slug: str) -> bool:
    """Guide the user through manual GitHub App creation.

    Args:
        slug: Suggested app name.

    Returns:
        True if the user provided valid credentials.
    """
    print("  Please create the app manually:\n")
    print(f"  1. Go to https://github.com/settings/apps/new")
    print(f"  2. GitHub App name: {slug}")
    print(f"  3. Homepage URL: https://github.com/otomus/arqitect")
    print(f"  4. Webhook:")
    print(f"     - Uncheck 'Active' (no webhook needed)")
    print(f"  5. Permissions → Repository permissions:")
    print(f"     - Contents: Read & Write")
    print(f"     - Metadata: Read-only")
    print(f"     - Pull requests: Read & Write")
    print(f"  6. Where can this app be installed?")
    print(f"     - Select 'Only on this account'")
    print(f"  7. Click 'Create GitHub App'")
    print(f"  8. On the app page, note the App ID at the top")
    print(f"  9. Scroll down to 'Private keys' (NOT client secrets)")
    print(f"     → Click 'Generate a private key'")
    print(f"     → This downloads a .pem file (starts with -----BEGIN RSA PRIVATE KEY-----)")
    print(f"  10. Enter the details below:\n")

    app_id = input("  App ID: ").strip()
    private_key = _read_private_key()
    key_path = _store_private_key(private_key)

    set_secret("github.app_id", app_id)
    set_secret("github.private_key_path", str(key_path))
    set_secret("github.app_slug", slug)
    print(f"\n  Private key saved to {key_path}")
    print(f"  GitHub App configured: {slug}")
    print(f"  Now install it on {COMMUNITY_REPO}:")
    print(f"  https://github.com/settings/apps/{slug}/installations/new")
    return True


def setup_github_app(app_name: str) -> bool:
    """Create a GitHub App and store its credentials in arqitect.yaml.

    Attempts the browser-based manifest flow first. If the browser flow
    fails or times out, falls back to manual setup instructions.

    Args:
        app_name: Display name for the app (e.g. ``arqitect-ob1``).

    Returns:
        True if setup completed successfully.
    """
    slug = app_name.lower().replace(" ", "-")

    print(f"\n  Creating GitHub App '{slug}'...")
    print("  This will open your browser for authorization.\n")

    try:
        if _setup_via_manifest(slug):
            return True
    except Exception as exc:
        logger.debug("[GITHUB] Manifest flow failed: %s", exc)
        print(f"  Browser-based setup failed: {exc}\n")

    try:
        return _setup_manually(slug)
    except Exception as exc:
        print(f"  Error during manual setup: {exc}")
        return False
