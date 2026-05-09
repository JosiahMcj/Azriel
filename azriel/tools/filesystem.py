"""filesystem tool -- sandboxed list / read / write.

Sandbox: writes always live inside ~/azriel-files/ on the host machine. Reads can
also follow explicit READ_MOUNTS -- symlinks placed inside the sandbox
that point at curated read-only reference dirs (e.g. the Missler
handbook PDFs). Path traversal (../) is rejected without following
symlinks at the resolve step, so the sandbox can't be escaped via a
relative path; legitimate symlinks the user has placed are honored
when the OS opens the file.

Three model-callable entry points are exposed via the registry:
  fs_list("dir")
  fs_read("relative/file.txt")
  fs_write("relative/file.txt|the file contents to save")
"""
import os
from pathlib import Path

SANDBOX = Path.home() / "azriel-files"
MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 500_000


def _ensure_sandbox():
    SANDBOX.mkdir(parents=True, exist_ok=True)
    # Use absolute (not resolve) so we don't pre-collapse symlinks the
    # user installed inside the sandbox as virtual mounts.
    return SANDBOX.absolute()


def _resolve(path_str: str) -> Path:
    base = _ensure_sandbox()
    raw = (SANDBOX / path_str.lstrip("/")).absolute()
    # Block traversal (..) without resolving symlinks. normpath collapses
    # '..' textually, so attempts to escape the sandbox prefix are rejected
    # but a symlink inside the sandbox to an external dir is allowed.
    p = Path(os.path.normpath(raw))
    base_s = str(base).rstrip(os.sep) + os.sep
    if str(p) != str(base) and not str(p).startswith(base_s):
        raise ValueError(f"path '{path_str}' escapes the sandbox")
    return p


def fs_list(path: str) -> str:
    if not isinstance(path, str):
        return "ERROR: fs_list expects a string path."
    try:
        p = _resolve(path or ".")
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: not found: {path}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    entries = []
    for child in sorted(p.iterdir()):
        rel = str(child.relative_to(SANDBOX.absolute()))
        if child.is_dir():
            entries.append(f" [dir] {rel}/")
        else:
            size = child.stat().st_size
            entries.append(f" [file] {rel} ({size:,} B)")
    if not entries:
        return f"(empty dir: {path})"
    return f"sandbox listing of '{path}':\n" + "\n".join(entries)


def fs_read(path: str) -> str:
    if not isinstance(path, str):
        return "ERROR: fs_read expects a string path."
    try:
        p = _resolve(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    size = p.stat().st_size
    if size > MAX_READ_BYTES:
        return f"ERROR: file too large ({size:,} B; max {MAX_READ_BYTES:,})."
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: read failed ({type(e).__name__}: {e})"


def fs_write(arg: str) -> str:
    if not isinstance(arg, str):
        return "ERROR: fs_write expects 'path|content'."
    if "|" not in arg:
        return "ERROR: format is 'path|content' (separator is the first '|')."
    path, content = arg.split("|", 1)
    path = path.strip()
    if not path:
        return "ERROR: empty path."
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return f"ERROR: content too large (max {MAX_WRITE_BYTES:,} bytes)."
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: write failed ({type(e).__name__}: {e})"
    return f"wrote {len(content):,} chars to {path}"


if __name__ == "__main__":
    import sys
    print(fs_list(sys.argv[1] if len(sys.argv) > 1 else "."))
