"""Check Markdown files for Markdown hygiene."""

from __future__ import annotations

import sys
from pathlib import Path
import re
import unicodedata


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = ascii_text.strip()
    ascii_text = re.sub(r"[#]+$", "", ascii_text).strip()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text


def collect_heading_ids(lines: list[str]) -> set[str]:
    ids: set[str] = set()
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            heading_text = match.group(2)
            ids.add(slugify(heading_text))
    return ids


def check_link_fragments(path: Path, lines: list[str], heading_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for idx, line in enumerate(lines, start=1):
        for _, target in LINK_RE.findall(line):
            if target.startswith("http://") or target.startswith("https://") or target.startswith("mailto:"):
                continue
            if "#" not in target:
                continue
            prefix, fragment = target.split("#", 1)
            if prefix and Path(prefix).name not in ("", path.name):
                continue
            if not fragment:
                continue
            if fragment not in heading_ids:
                errors.append(
                    f"{path}:{idx}: unknown link fragment '#{fragment}'"
                )
    return errors


def check_fenced_blocks(path: Path, lines: list[str]) -> list[str]:
    errors: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].lstrip()
        if line.startswith("```"):
            fence_start = idx
            if fence_start > 0 and lines[fence_start - 1].strip():
                errors.append(
                    f"{path}:{fence_start + 1}: missing blank line before fenced block"
                )

            idx += 1
            while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
                idx += 1

            if idx >= len(lines):
                break  # unclosed fence; other tooling will complain

            fence_end = idx
            next_line_index = fence_end + 1
            if next_line_index < len(lines) and lines[next_line_index].strip():
                errors.append(
                    f"{path}:{fence_end + 1}: missing blank line after fenced block"
                )
        idx += 1
    return errors


def check_file(path: Path) -> list[str]:
    """Return list of human-readable errors for the given Markdown file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    heading_ids = collect_heading_ids(lines)
    errors: list[str] = []
    errors.extend(check_fenced_blocks(path, lines))
    errors.extend(check_link_fragments(path, lines, heading_ids))
    return errors


def main() -> int:
    files = [Path(arg) for arg in sys.argv[1:]] or list(Path.cwd().glob("**/*.md"))
    problems: list[str] = []
    for file_path in files:
        if file_path.is_file():
            problems.extend(check_file(file_path))

    if problems:
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
