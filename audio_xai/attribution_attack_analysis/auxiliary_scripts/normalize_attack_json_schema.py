"""
Normalize JSON result schemas for attribution attack analysis.

This script maps x-shift field names to the canonical names used by the
analysis pipeline:

    cos_orig_adv          -> cos_sim
    top10_overlap_orig    -> top10_overlap

It can process one or more folders recursively. By default, it only fills
missing canonical fields and preserves existing values. Use --overwrite to
force canonical values to be replaced by their alias values.

Examples
--------
Dry run:
    python normalize_attack_json_schema.py /path/to/attack_3 --dry-run

Modify files and create .bak backups:
    python normalize_attack_json_schema.py /path/to/attack_3

Modify files without backups:
    python normalize_attack_json_schema.py /path/to/attack_3 --no-backup

Process many folders:
    python normalize_attack_json_schema.py data/model_a/Attack_3 data/model_b/Attack_3

Force overwrite canonical fields from alias fields:
    python normalize_attack_json_schema.py /path/to/attack_3 --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALIASES: dict[str, str] = {
    "cos_orig_adv": "cos_sim",
    "top10_overlap_orig": "top10_overlap",
}


@dataclass
class FileResult:
    path: Path
    changed: bool
    updates: list[str]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize x-shift JSON field names to canonical names used by "
            "the attribution attack analysis pipeline."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more JSON files or directories containing JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite canonical fields even when they already exist. "
            "By default the script only fills missing/null canonical fields."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files before modifying JSON files.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation used when saving modified files. Default: 2.",
    )
    return parser.parse_args()


def iter_json_files(paths: list[Path]) -> list[Path]:
    json_files: list[Path] = []

    for input_path in paths:
        path = input_path.expanduser().resolve()

        if not path.exists():
            print(f"[WARN] Path does not exist: {path}", file=sys.stderr)
            continue

        if path.is_file():
            if path.suffix.lower() == ".json":
                json_files.append(path)
            else:
                print(f"[WARN] Skipping non-JSON file: {path}", file=sys.stderr)
            continue

        json_files.extend(sorted(path.rglob("*.json")))

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for file_path in json_files:
        if file_path not in seen:
            unique.append(file_path)
            seen.add(file_path)
    return unique


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any, indent: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def normalize_record(record: dict[str, Any], overwrite: bool) -> list[str]:
    updates: list[str] = []

    for source_col, target_col in ALIASES.items():
        if source_col not in record:
            continue

        source_value = record[source_col]
        target_missing = target_col not in record or record[target_col] is None

        if target_missing:
            record[target_col] = source_value
            updates.append(f"added {target_col} from {source_col}")
        elif overwrite and record[target_col] != source_value:
            old_value = record[target_col]
            record[target_col] = source_value
            updates.append(
                f"overwrote {target_col}: {old_value!r} -> {source_value!r} from {source_col}"
            )

    return updates


def normalize_json_data(data: Any, overwrite: bool) -> list[str]:
    """
    Normalize supported JSON shapes:
    - a single object: { ... }
    - a list of objects: [{ ... }, { ... }]
    - an object containing a list under common keys such as results/items/samples/data
    """
    updates: list[str] = []

    if isinstance(data, dict):
        updates.extend(normalize_record(data, overwrite))

        for key in ("results", "items", "samples", "data"):
            value = data.get(key)
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        item_updates = normalize_record(item, overwrite)
                        updates.extend([f"{key}[{idx}]: {u}" for u in item_updates])

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                item_updates = normalize_record(item, overwrite)
                updates.extend([f"[{idx}]: {u}" for u in item_updates])

    return updates


def process_file(
    path: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    backup: bool,
    indent: int,
) -> FileResult:
    try:
        data = load_json(path)
        updates = normalize_json_data(data, overwrite=overwrite)

        if not updates:
            return FileResult(path=path, changed=False, updates=[])

        if not dry_run:
            if backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                if not backup_path.exists():
                    shutil.copy2(path, backup_path)
            save_json(path, data, indent=indent)

        return FileResult(path=path, changed=True, updates=updates)
    except Exception as exc:  # noqa: BLE001 - CLI tool should report and continue.
        return FileResult(path=path, changed=False, updates=[], error=str(exc))


def main() -> int:
    args = parse_args()
    json_files = iter_json_files(args.paths)

    if not json_files:
        print("No JSON files found.")
        return 1

    results = [
        process_file(
            path,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            backup=not args.no_backup,
            indent=args.indent,
        )
        for path in json_files
    ]

    changed = [r for r in results if r.changed]
    unchanged = [r for r in results if not r.changed and not r.error]
    errors = [r for r in results if r.error]

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"Mode: {mode}")
    print(f"Scanned JSON files: {len(results)}")
    print(f"Changed files: {len(changed)}")
    print(f"Unchanged files: {len(unchanged)}")
    print(f"Errors: {len(errors)}")

    if changed:
        print("\nChanged files:")
        for result in changed:
            print(f"- {result.path}")
            for update in result.updates[:8]:
                print(f"    * {update}")
            if len(result.updates) > 8:
                print(f"    * ... {len(result.updates) - 8} more updates")

    if errors:
        print("\nErrors:", file=sys.stderr)
        for result in errors:
            print(f"- {result.path}: {result.error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
