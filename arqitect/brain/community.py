"""Community integration — sync nerves, tools, and adapters from arqitect-community.

Fetches the manifest from https://github.com/otomus/sentient-community and caches
nerve bundles, tool implementations, and adapter metadata locally.

Called during brain bootstrap to seed the server with community content.
"""

import json
import os
import shutil
import urllib.request
import urllib.error

from arqitect.config.loader import get_config, get_project_root, get_mcp_tools_dir

COMMUNITY_REPO = "otomus/sentient-community"
COMMUNITY_RAW_URL = f"https://raw.githubusercontent.com/{COMMUNITY_REPO}/main"

# Tags that lock a nerve to a specific environment
ENV_EXCLUSIVE_TAGS = {"iot", "desktop"}


def _cache_dir() -> str:
    return os.path.join(str(get_project_root()), ".community", "cache")


def _manifest_path() -> str:
    return os.path.join(_cache_dir(), "manifest.json")


def _load_cached_manifest() -> dict | None:
    path = _manifest_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sync_manifest() -> dict | None:
    """Fetch the latest manifest from GitHub and cache it locally."""
    cache = _cache_dir()
    os.makedirs(cache, exist_ok=True)
    url = f"{COMMUNITY_RAW_URL}/manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        with open(_manifest_path(), "w") as f:
            json.dump(data, f, indent=2)
        return data
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        print(f"[COMMUNITY] Failed to sync manifest: {e}")
        return None


def sync_nerve_bundle(name: str) -> str | None:
    """Download a nerve bundle from the community repo to local cache.

    Downloads bundle.json, test_cases.json, tool implementations, and
    per-size-class context.json/meta.json files for our active model.
    """
    bundle_dir = os.path.join(_cache_dir(), "nerves", name)
    os.makedirs(bundle_dir, exist_ok=True)

    # Core files
    for fname in ("bundle.json", "test_cases.json"):
        url = f"{COMMUNITY_RAW_URL}/nerves/{name}/{fname}"
        local_path = os.path.join(bundle_dir, fname)
        try:
            urllib.request.urlretrieve(url, local_path)
        except urllib.error.HTTPError:
            if fname == "bundle.json":
                shutil.rmtree(bundle_dir, ignore_errors=True)
                return None

    bundle_path = os.path.join(bundle_dir, "bundle.json")
    if not os.path.exists(bundle_path):
        return None

    try:
        with open(bundle_path) as f:
            bundle = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Download tool implementations
    for tool in bundle.get("tools", []):
        for _lang, impl_path in tool.get("implementations", {}).items():
            url = f"{COMMUNITY_RAW_URL}/nerves/{name}/{impl_path}"
            local_dir = os.path.join(bundle_dir, os.path.dirname(impl_path))
            os.makedirs(local_dir, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, os.path.join(bundle_dir, impl_path))
            except urllib.error.HTTPError:
                pass

    # Download per-size-class context.json + meta.json for our model
    _sync_nerve_adapter_files(name, bundle, bundle_dir)

    return bundle_dir


def _sync_nerve_adapter_files(name: str, bundle: dict, bundle_dir: str):
    """Download context.json and meta.json for our active model's size class.

    Mirrors the adapter directory structure:
        {size_class}/context.json, {size_class}/meta.json
        {size_class}/{model_slug}/context.json, {size_class}/{model_slug}/meta.json
        {size_class}/{model_slug}/adapter.gguf (LoRA)
    """
    try:
        from arqitect.brain.adapters import get_active_variant, get_model_name_for_role
        role = bundle.get("role", "tool")
        size_class = get_active_variant(role)
        model_slug = get_model_name_for_role(role)
    except (ImportError, ValueError):
        return

    if not size_class:
        return

    # Size-class level files
    for fname in ("context.json", "meta.json"):
        url = f"{COMMUNITY_RAW_URL}/nerves/{name}/{size_class}/{fname}"
        local_dir = os.path.join(bundle_dir, size_class)
        os.makedirs(local_dir, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, os.path.join(local_dir, fname))
        except urllib.error.HTTPError:
            pass

    if not model_slug:
        return

    # Model-specific files
    for fname in ("context.json", "meta.json"):
        url = f"{COMMUNITY_RAW_URL}/nerves/{name}/{size_class}/{model_slug}/{fname}"
        local_dir = os.path.join(bundle_dir, size_class, model_slug)
        os.makedirs(local_dir, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, os.path.join(local_dir, fname))
        except urllib.error.HTTPError:
            pass

    # LoRA adapter
    adapter_url = f"{COMMUNITY_RAW_URL}/nerves/{name}/{size_class}/{model_slug}/adapter.gguf"
    adapter_dir = os.path.join(bundle_dir, size_class, model_slug)
    os.makedirs(adapter_dir, exist_ok=True)
    try:
        urllib.request.urlretrieve(adapter_url, os.path.join(adapter_dir, "adapter.gguf"))
        print(f"[COMMUNITY] Downloaded LoRA adapter for {name}/{size_class}/{model_slug}")
    except urllib.error.HTTPError:
        pass


def find_community_bundle(nerve_name: str) -> dict | None:
    """Check if the community has a bundle matching this nerve name."""
    bundle_path = os.path.join(_cache_dir(), "nerves", nerve_name, "bundle.json")
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    manifest = _load_cached_manifest()
    if manifest and nerve_name in manifest.get("nerves", {}):
        bundle_dir = sync_nerve_bundle(nerve_name)
        if bundle_dir:
            bundle_path = os.path.join(bundle_dir, "bundle.json")
            try:
                with open(bundle_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    return None


def apply_community_bundle(nerve_name: str, bundle: dict, cold_memory) -> dict:
    """Apply a community bundle's identity to a nerve being synthesized.

    Reads system_prompt and examples from per-size-class context.json files
    (same layout as adapters/). Falls back to bundle.default for old-format bundles.
    """
    from arqitect.brain.routing import validate_nerve_role
    mcp_tools_dir = str(get_mcp_tools_dir())
    description = bundle.get("description", "")
    role = validate_nerve_role(bundle.get("role", "tool"))

    # Resolve prompt from context.json files (new structure)
    system_prompt, examples = _resolve_bundle_prompt(nerve_name, role)

    # Fallback to old bundle.default for bundles not yet migrated
    if not system_prompt:
        default = bundle.get("default", {})
        system_prompt = default.get("system_prompt", "")
        examples = default.get("examples", [])

    cold_memory.register_nerve_rich(
        nerve_name, description, system_prompt,
        json.dumps(examples), role=role, origin="community",
    )

    bundle_dir = os.path.join(_cache_dir(), "nerves", nerve_name)
    installed_tools = _install_bundle_tools(nerve_name, bundle, bundle_dir, mcp_tools_dir, cold_memory)
    _install_bundle_tests(nerve_name, bundle_dir, cold_memory)
    _install_bundle_lora(nerve_name, role, bundle_dir)

    return {
        "description": description,
        "role": role,
        "system_prompt": system_prompt,
        "examples": examples,
        "tools": installed_tools,
    }


def _resolve_bundle_prompt(nerve_name: str, role: str) -> tuple[str, list]:
    """Read system_prompt and examples from cached context.json for this nerve."""
    try:
        from arqitect.brain.adapters import resolve_nerve_prompt
        ctx = resolve_nerve_prompt(nerve_name, role)
        if ctx:
            return (
                ctx.get("system_prompt", ""),
                ctx.get("few_shot_examples", []),
            )
    except (ImportError, ValueError):
        pass
    return "", []


def _install_bundle_tools(nerve_name: str, bundle: dict, bundle_dir: str,
                          mcp_tools_dir: str, cold_memory) -> list[str]:
    """Copy tool implementations from bundle cache to mcp_tools/ and wire them.

    Only handles directory-based tools with tool.json.

    Args:
        nerve_name: Name of the nerve owning these tools.
        bundle: Parsed bundle.json dict.
        bundle_dir: Path to the cached bundle directory.
        mcp_tools_dir: Path to the mcp_tools/ directory.
        cold_memory: Cold memory instance.

    Returns:
        List of installed tool names.
    """
    installed = []
    for tool in bundle.get("tools", []):
        tool_name = tool["name"]

        # Install directory-based tool from bundle
        tool_json_src = os.path.join(bundle_dir, "tools", tool_name, "tool.json")
        if os.path.isfile(tool_json_src):
            _install_tool_directory(tool_name, bundle_dir, mcp_tools_dir)

        cold_memory.add_nerve_tool(nerve_name, tool_name)
        installed.append(tool_name)
    return installed


def _install_tool_directory(tool_name: str, bundle_dir: str,
                            mcp_tools_dir: str) -> None:
    """Copy a directory-based tool from bundle cache to mcp_tools/.

    Creates mcp_tools/{tool_name}/ with tool.json, entry point, and dep files.

    Args:
        tool_name: Name of the tool.
        bundle_dir: Path to the cached bundle directory.
        mcp_tools_dir: Path to the mcp_tools/ directory.
    """
    src_dir = os.path.join(bundle_dir, "tools", tool_name)
    dst_dir = os.path.join(mcp_tools_dir, tool_name)

    if os.path.isdir(dst_dir):
        return  # Already installed

    shutil.copytree(src_dir, dst_dir)
    print(f"[COMMUNITY] Installed tool directory: {tool_name}")


def _install_bundle_tests(nerve_name: str, bundle_dir: str, cold_memory):
    """Load test_cases.json from bundle cache into cold memory."""
    tests_path = os.path.join(bundle_dir, "test_cases.json")
    if not os.path.exists(tests_path):
        return
    try:
        with open(tests_path) as f:
            tests = json.load(f)
        cold_memory.set_test_bank(nerve_name, tests)
    except (json.JSONDecodeError, OSError):
        pass


def _install_bundle_lora(nerve_name: str, role: str, bundle_dir: str):
    """Copy LoRA adapter from bundle cache to the nerve's adapter directory."""
    try:
        from arqitect.brain.adapters import get_active_variant, get_model_name_for_role
        from arqitect.config.loader import get_nerves_dir
        size_class = get_active_variant(role)
        model_slug = get_model_name_for_role(role)
    except (ImportError, ValueError):
        return

    if not size_class or not model_slug:
        return

    adapter_src = os.path.join(bundle_dir, size_class, model_slug, "adapter.gguf")
    if not os.path.isfile(adapter_src):
        return

    adapter_dest_dir = os.path.join(str(get_nerves_dir()), nerve_name, "adapter")
    os.makedirs(adapter_dest_dir, exist_ok=True)
    dest = os.path.join(adapter_dest_dir, "adapter.gguf")
    if not os.path.exists(dest):
        shutil.copy2(adapter_src, dest)
        print(f"[COMMUNITY] Installed LoRA adapter for {nerve_name}")


def seed_tools() -> int:
    """Download any community tools missing from mcp_tools/.

    Reads the cached manifest and installs directory-based tool implementations
    that don't already exist locally. Environments are NOT built at startup
    — they're built during dream state or on first call.

    Called at brain startup so pre-seeded tools are available when nerves invoke them.

    Returns:
        Number of tools installed.
    """
    manifest = _load_cached_manifest()
    if not manifest:
        return 0

    tools = manifest.get("tools", {})
    if not tools:
        return 0

    mcp_tools_dir = str(get_mcp_tools_dir())
    os.makedirs(mcp_tools_dir, exist_ok=True)
    installed = 0

    for name, info in tools.items():
        if "version" in info and "files" in info:
            if _seed_tool_directory(name, info, mcp_tools_dir):
                installed += 1

    if installed:
        print(f"[COMMUNITY] Seeded {installed} tool(s) from community")
    return installed


def _seed_tool_directory(name: str, info: dict, mcp_tools_dir: str) -> bool:
    """Download a directory-based tool from the community repo.

    Downloads all files listed in the manifest entry into mcp_tools/{name}/.
    Marks the tool as needs_build — environment creation is deferred to dream state.

    Args:
        name: Tool name.
        info: Manifest tool entry with version, files, runtime fields.
        mcp_tools_dir: Path to the mcp_tools/ directory.

    Returns:
        True if the tool was downloaded (new or updated).
    """
    tool_dir = os.path.join(mcp_tools_dir, name)
    version_file = os.path.join(tool_dir, ".env_version")
    remote_version = info.get("version", "0.0.0")

    # Skip if already at this version
    if os.path.isdir(tool_dir) and os.path.isfile(version_file):
        try:
            with open(version_file) as f:
                local_version = f.read().strip()
            if local_version == remote_version:
                return False
        except OSError:
            pass

    os.makedirs(tool_dir, exist_ok=True)

    files = info.get("files", ["tool.json"])
    downloaded = False
    for fname in files:
        url = f"{COMMUNITY_RAW_URL}/tools/{name}/{fname}"
        dest = os.path.join(tool_dir, fname)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded = True
        except urllib.error.HTTPError:
            if fname == "tool.json":
                # tool.json is required — abort
                shutil.rmtree(tool_dir, ignore_errors=True)
                return False

    if downloaded:
        # Mark as needs_build — no env_version yet means dream state will build it
        needs_build_marker = os.path.join(tool_dir, ".needs_build")
        with open(needs_build_marker, "w") as f:
            f.write(remote_version)
        print(f"[COMMUNITY] Seeded tool directory: {name} (v{remote_version}, build deferred)")
        return True

    return False


def _matches_environment(tags: list[str], environment: str) -> bool:
    """Check if a nerve's tags are compatible with the configured environment.

    A nerve with no environment-exclusive tags is universal (matches all).
    A nerve tagged with an exclusive tag (e.g. "iot", "desktop") only matches
    when the configured environment equals that tag.

    Args:
        tags: Tag list from the manifest nerve entry.
        environment: The configured environment (e.g. "server", "iot", "desktop").

    Returns:
        True if the nerve should be seeded in this environment.
    """
    exclusive = ENV_EXCLUSIVE_TAGS & set(tags)
    if not exclusive:
        return True
    return environment in exclusive


def seed_nerves() -> int:
    """Bootstrap community nerves from manifest metadata (no network).

    Registers nerves in cold memory and writes nerve.py files using only
    the manifest's inline metadata (description, role, tools). This is
    instant regardless of how many nerves the community has.

    Full bundle sync (system_prompt, examples, adapters) is deferred to
    dreamstate or first invocation via hydrate_nerve_bundle().

    Skips nerves that already have a nerve.py on disk (but rewires tools).

    Returns:
        Number of nerves bootstrapped.
    """
    manifest = _load_cached_manifest()
    if not manifest:
        return 0

    nerves = manifest.get("nerves", {})
    if not nerves:
        return 0

    from arqitect.brain.config import NERVES_DIR, mem

    environment = get_config("environment", "server")

    bootstrapped = 0
    pruned = 0
    for name, info in nerves.items():
        tags = info.get("tags", [])
        nerve_dir = os.path.join(NERVES_DIR, name)
        nerve_path = os.path.join(nerve_dir, "nerve.py")

        if not _matches_environment(tags, environment):
            # Remove nerves from previous seeds that no longer match
            if os.path.isfile(nerve_path):
                shutil.rmtree(nerve_dir, ignore_errors=True)
                mem.cold.delete_nerve(name)
                pruned += 1
            continue

        if os.path.isfile(nerve_path):
            _rewire_nerve_tools(name, info, mem.cold)
            continue

        _bootstrap_nerve_lightweight(name, info, NERVES_DIR, mem.cold)
        bootstrapped += 1

    if pruned:
        print(f"[COMMUNITY] Pruned {pruned} nerve(s) not matching environment '{environment}'")
    if bootstrapped:
        print(f"[COMMUNITY] Registered {bootstrapped} nerve(s) from community (bundles deferred)")
    return bootstrapped


def _bootstrap_nerve_lightweight(name: str, manifest_info: dict, nerves_dir: str,
                                  cold_memory) -> None:
    """Register a community nerve using only manifest metadata — no network IO.

    Creates nerve.py on disk and registers description/role/tools in cold memory.
    The full bundle (system_prompt, examples, adapters) is fetched later by
    hydrate_nerve_bundle() during dreamstate or first invocation.

    Args:
        name: Nerve name from the manifest.
        manifest_info: Nerve entry from manifest (description, role, tools).
        nerves_dir: Path to the nerves directory.
        cold_memory: Cold memory instance for registration.
    """
    from arqitect.brain.nerve_template import NERVE_TEMPLATE
    from arqitect.brain.routing import classify_nerve_role

    description = manifest_info.get("description", name)
    role = manifest_info.get("role")
    if not role:
        role = classify_nerve_role(name, description)

    _write_nerve_file(name, role, description, nerves_dir, NERVE_TEMPLATE)
    cold_memory.register_nerve_rich(name, description, "", "[]", role=role, origin="community")
    _rewire_nerve_tools(name, manifest_info, cold_memory)

    print(f"[COMMUNITY] Registered nerve: {name} (role={role})")


def hydrate_nerve_bundle(name: str) -> bool:
    """Fetch and apply the full community bundle for a nerve.

    Downloads bundle.json, system_prompt, examples, tool implementations,
    and LoRA adapters from the community repo. Called lazily from dreamstate
    or on first nerve invocation.

    Args:
        name: Nerve name to hydrate.

    Returns:
        True if bundle was applied, False if unavailable.
    """
    from arqitect.brain.config import mem

    bundle = _load_nerve_bundle(name)
    if not bundle:
        return False

    apply_community_bundle(name, bundle, mem.cold)
    print(f"[COMMUNITY] Hydrated nerve bundle: {name}")
    return True


def _bootstrap_nerve(name: str, manifest_info: dict, nerves_dir: str,
                     cold_memory) -> None:
    """Create a single community nerve on disk and apply its bundle.

    Full bootstrap with network IO — used by sync_all() and manual seeding.
    For startup, use _bootstrap_nerve_lightweight() instead.

    Args:
        name: Nerve name from the manifest.
        manifest_info: Nerve entry from manifest (description, role, tools).
        nerves_dir: Path to the nerves directory.
        cold_memory: Cold memory instance for registration.
    """
    from arqitect.brain.nerve_template import NERVE_TEMPLATE
    from arqitect.brain.routing import classify_nerve_role

    bundle = _load_nerve_bundle(name)
    description = manifest_info.get("description", name)
    role = manifest_info.get("role") or (bundle.get("role") if bundle else None)
    if not role:
        role = classify_nerve_role(name, description)

    _write_nerve_file(name, role, description, nerves_dir, NERVE_TEMPLATE)

    if bundle:
        apply_community_bundle(name, bundle, cold_memory)
    else:
        cold_memory.register_nerve_rich(name, description, "", "[]", role=role, origin="community")
        _rewire_nerve_tools(name, manifest_info, cold_memory)

    print(f"[COMMUNITY] Bootstrapped nerve: {name} (role={role})")


def _load_nerve_bundle(name: str) -> dict | None:
    """Sync and load a nerve bundle from the community repo.

    Returns:
        Parsed bundle dict, or None if unavailable.
    """
    bundle_dir = sync_nerve_bundle(name)
    if not bundle_dir:
        return None
    bundle_path = os.path.join(bundle_dir, "bundle.json")
    try:
        with open(bundle_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_nerve_file(name: str, role: str, description: str,
                      nerves_dir: str, template: str) -> None:
    """Write nerve.py from template into the nerve directory.

    Args:
        name: Nerve name (used as directory name and template variable).
        role: Nerve role (tool/creative/code).
        description: Domain-level nerve description.
        nerves_dir: Parent directory for all nerves.
        template: Nerve template string with {{placeholders}}.
    """
    nerve_dir = os.path.join(nerves_dir, name)
    os.makedirs(nerve_dir, exist_ok=True)
    nerve_path = os.path.join(nerve_dir, "nerve.py")
    with open(nerve_path, "w") as f:
        f.write(template.replace("{{NERVE_NAME}}", name)
                        .replace("{{NERVE_ROLE}}", role)
                        .replace("{{DESCRIPTION}}", description))
    os.chmod(nerve_path, 0o755)


def _rewire_nerve_tools(name: str, manifest_info: dict, cold_memory) -> None:
    """Ensure a nerve's tool wiring matches the community manifest declaration.

    Adds any tools declared in the manifest that are missing from cold memory.
    Never removes existing tools — only fills gaps.

    Args:
        name: Nerve name.
        manifest_info: Nerve entry from manifest containing a 'tools' list.
        cold_memory: Cold memory instance for tool registration.
    """
    declared_tools = manifest_info.get("tools", [])
    if not declared_tools:
        return

    existing_tools = set(cold_memory.get_nerve_tools(name))
    for tool_name in declared_tools:
        if tool_name not in existing_tools:
            cold_memory.add_nerve_tool(name, tool_name)
            print(f"[COMMUNITY] Wired tool '{tool_name}' -> nerve '{name}'")


def seed_dependencies() -> bool:
    """Fetch and run the community's seed_dependencies.py to install required packages.

    Downloads scripts/seed_dependencies.py from the community repo and runs it
    with --install --tools-dir pointing to the community's cached mcp_tools.

    Returns:
        True if dependencies were installed successfully or none were needed.
    """
    import subprocess
    import sys

    cache = _cache_dir()
    os.makedirs(cache, exist_ok=True)

    script_url = f"{COMMUNITY_RAW_URL}/scripts/seed_dependencies.py"
    script_path = os.path.join(cache, "seed_dependencies.py")

    try:
        urllib.request.urlretrieve(script_url, script_path)
    except urllib.error.HTTPError as e:
        print(f"[COMMUNITY] No seed_dependencies.py in community repo: {e}")
        return False

    # The community script expects a tools dir with meta.json per tool.
    # Point it at the community cache or the server's own mcp_tools.
    tools_dir = str(get_mcp_tools_dir())

    try:
        result = subprocess.run(
            [sys.executable, script_path, "--install", "--tools-dir", tools_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.stdout.strip():
            print(f"[COMMUNITY] {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"[COMMUNITY] seed_dependencies.py failed: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[COMMUNITY] seed_dependencies.py timed out")
        return False
    except Exception as e:
        print(f"[COMMUNITY] Failed to run seed_dependencies.py: {e}")
        return False


def sync_all() -> dict:
    """Full sync: pull manifest + all nerve bundles.

    Called during brain bootstrap or dream state.
    """
    print("[COMMUNITY] Syncing manifest...")
    manifest = sync_manifest()
    if not manifest:
        return {"error": "Could not fetch manifest"}

    nerves = manifest.get("nerves", {})
    adapters = manifest.get("adapters", {})
    connectors = manifest.get("connectors", {})

    synced_nerves = 0
    for name in nerves:
        print(f"[COMMUNITY] Syncing nerve: {name}")
        if sync_nerve_bundle(name):
            synced_nerves += 1

    stats = {
        "nerves_available": len(nerves),
        "nerves_synced": synced_nerves,
        "adapters_available": len(adapters),
        "connectors_available": len(connectors),
    }
    print(f"[COMMUNITY] Sync complete: {synced_nerves}/{len(nerves)} nerves cached, "
          f"{len(adapters)} adapters, {len(connectors)} connectors in manifest")
    return stats
