"""
Security helpers — path traversal protection, prompt sanitization,
safe subprocess argument construction.

CRITICAL: All user input that flows into subprocess.run() must go through
validate_path() and validate_prompt(). Never use os.system() or shell=True.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


# ---------- Regex patterns ----------
# Allow: alphanumerics, underscore, hyphen, dot, forward slash, backslash, spaces, CJK
SAFE_PATH_RE = re.compile(r"^[a-zA-Z0-9_\-\./\\\s\u4e00-\u9fff\u0400-\u04FF]+$")

# Filename: alphanumerics, underscore, hyphen, dot — NO slashes, NO spaces (for output filenames)
SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-.]+$")

# Prompt: any printable character except null byte — but limit length
PROMPT_MAX_LEN = 8192
NEGATIVE_PROMPT_MAX_LEN = 2048


class SecurityError(ValueError):
    """Raised when user input fails validation."""


# ---------- Path validation ----------
def validate_path(p: str | Path, base_dir: Path, must_exist: bool = False) -> Path:
    """Validate that a user-supplied path is safe and within base_dir.

    Security strategy:
      1. Reject null bytes (path injection attack on Windows / Linux)
      2. Reject control characters (ord < 32 except newline/tab — not valid in paths)
      3. Resolve to absolute path via Path.resolve()
      4. Verify resolved path is within base_dir (path traversal protection)

    The character-level regex is intentionally permissive — real protection
    comes from the .resolve() + .relative_to() check below. This handles
    Windows drive letters (C:\\), UNC paths (\\\\server\\share), forward slashes,
    backslashes, spaces, CJK, Cyrillic, etc. without false positives.

    Args:
        p: User-supplied path string (may be relative or absolute)
        base_dir: Directory that the resolved path must be inside of
        must_exist: If True, raise if the path doesn't exist on disk

    Returns:
        Resolved absolute Path

    Raises:
        SecurityError: If path contains null bytes, control chars,
                       or escapes base_dir
    """
    if p is None:
        raise SecurityError("Path is None")

    # Decode URL-encoded sequences (e.g., %2e%2e%2f → ../)
    if isinstance(p, str):
        p = unquote(p)

    p_str = str(p)

    # Reject empty paths
    if not p_str.strip():
        raise SecurityError("Path is empty")

    # Reject null bytes (path injection attack)
    if "\x00" in p_str:
        raise SecurityError("Path contains null byte")

    # Reject other control characters (ord < 32 except tab/newline, plus DEL=127)
    # These should never appear in legitimate filesystem paths.
    bad_chars = [c for c in p_str if (ord(c) < 32 and c not in "\t\n") or ord(c) == 127]
    if bad_chars:
        raise SecurityError(
            f"Path contains control characters: {bad_chars!r} in {p_str!r}"
        )

    # Resolve to absolute path
    base_dir_resolved = base_dir.resolve()
    try:
        if Path(p).is_absolute():
            resolved = Path(p).resolve()
        else:
            resolved = (base_dir / p).resolve()
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Failed to resolve path {p!r}: {e}")

    # Verify resolved path is within base_dir (path traversal protection)
    try:
        resolved.relative_to(base_dir_resolved)
    except ValueError:
        raise SecurityError(
            f"Path escapes base directory: {p!r} resolves to {resolved} "
            f"which is outside {base_dir_resolved}"
        )

    # Optional: check existence
    if must_exist and not resolved.exists():
        raise SecurityError(f"Path does not exist: {resolved}")

    return resolved


def validate_filename(name: str, max_length: int = 255) -> str:
    """Validate a single filename (no path separators allowed).

    Used for output filenames like 'my_image.png'.
    """
    if not name or not name.strip():
        raise SecurityError("Filename is empty")

    if len(name) > max_length:
        raise SecurityError(f"Filename too long: {len(name)} > {max_length}")

    if "\x00" in name:
        raise SecurityError("Filename contains null byte")

    if not SAFE_FILENAME_RE.match(name):
        raise SecurityError(
            f"Invalid filename: {name!r}. "
            f"Allowed: alphanumerics, _, -, . (no slashes, no spaces)."
        )

    # Reject Windows reserved names
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    stem = name.split(".")[0].upper()
    if stem in reserved:
        raise SecurityError(f"Filename uses reserved Windows name: {name!r}")

    return name


# ---------- Prompt validation ----------
def validate_prompt(prompt: str, max_length: int = PROMPT_MAX_LEN) -> str:
    """Sanitize a user-supplied prompt.

    - Strips control characters (except newline and tab)
    - Limits length
    - Returns the cleaned prompt

    NOTE: We do NOT escape shell metacharacters here because we always use
    subprocess.run() with a list (never shell=True). The prompt is passed
    as a single list element, so it cannot break out of argv.
    """
    if prompt is None:
        return ""

    if not isinstance(prompt, str):
        raise SecurityError(f"Prompt must be a string, got {type(prompt).__name__}")

    if len(prompt) > max_length:
        raise SecurityError(
            f"Prompt too long: {len(prompt)} > {max_length} characters"
        )

    # Remove control characters except newline (\n) and tab (\t)
    # Ord 0-31 are control chars; 127 is DEL; 128+ are printable
    cleaned = "".join(
        c for c in prompt
        if c == "\n" or c == "\t" or ord(c) >= 32 and ord(c) != 127
    )

    return cleaned.strip()


def validate_negative_prompt(prompt: str) -> str:
    """Like validate_prompt but with smaller max length."""
    return validate_prompt(prompt, max_length=NEGATIVE_PROMPT_MAX_LEN)


# ---------- Numeric validation ----------
def validate_int(value, name: str, min_val: int = 0, max_val: int = 2**31 - 1) -> int:
    """Validate that a value is an integer within bounds."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise SecurityError(f"{name} must be an integer, got {value!r}")

    if v < min_val or v > max_val:
        raise SecurityError(f"{name} out of range: {v} (must be {min_val}..{max_val})")

    return v


def validate_float(value, name: str, min_val: float = 0.0, max_val: float = 1e9) -> float:
    """Validate that a value is a float within bounds."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise SecurityError(f"{name} must be a number, got {value!r}")

    if v < min_val or v > max_val:
        raise SecurityError(f"{name} out of range: {v} (must be {min_val}..{max_val})")

    return v


def validate_resolution(width: int, height: int, min_px: int = 256, max_px: int = 4096) -> tuple[int, int]:
    """Validate image resolution. Both dimensions must be in [min_px, max_px]
    and ideally multiples of 8 (diffusers convention)."""
    w = validate_int(width, "width", min_px, max_px)
    h = validate_int(height, "height", min_px, max_px)
    # Snap to multiple of 8 (QuantFunc / diffusers convention)
    w = (w // 8) * 8
    h = (h // 8) * 8
    if w < min_px:
        w = min_px
    if h < min_px:
        h = min_px
    return w, h


# ---------- Subprocess argument safety ----------
def safe_subprocess_args(args: list[str]) -> list[str]:
    """Validate that all subprocess arguments are safe strings.

    Args:
        args: List of strings to pass to subprocess.run()

    Returns:
        The same list, validated.

    Raises:
        SecurityError: If any arg is not a string or contains null bytes.
    """
    if not isinstance(args, list):
        raise SecurityError(f"subprocess args must be a list, got {type(args).__name__}")

    if not args:
        raise SecurityError("subprocess args list is empty")

    safe = []
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            # Allow Path objects — convert to string
            if isinstance(arg, Path):
                arg = str(arg)
            else:
                raise SecurityError(
                    f"subprocess arg[{i}] must be a string, got {type(arg).__name__}: {arg!r}"
                )
        if "\x00" in arg:
            raise SecurityError(f"subprocess arg[{i}] contains null byte: {arg!r}")
        safe.append(arg)

    return safe
