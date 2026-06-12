import argparse
import json
from pathlib import Path

KEY_RENAMES = {
    "lrp_cos_sim": "cos_sim",
    "lrp_top10_overlap": "top10_overlap",
}


def rename_keys(obj):
    """
    Rekurencyjnie zmienia nazwy kluczy w dict/list,
    nie zmieniając wartości.
    """
    if isinstance(obj, dict):
        new_obj = {}

        for key, value in obj.items():
            new_key = KEY_RENAMES.get(key, key)

            if new_key in new_obj:
                raise ValueError(f"Kolizja kluczy: po zmianie '{key}' na '{new_key}' klucz już istnieje w tym samym obiekcie JSON.")

            new_obj[new_key] = rename_keys(value)

        return new_obj

    if isinstance(obj, list):
        return [rename_keys(item) for item in obj]

    return obj


def process_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated = rename_keys(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", nargs="?", default=Path.cwd(), type=Path)
    return parser.parse_args()


def main():
    args = _parse_args()
    json_files = list(args.root_dir.rglob("*.json"))

    print(f"Znaleziono plików JSON: {len(json_files)}")

    changed = 0
    failed = 0

    for json_path in json_files:
        try:
            process_json_file(json_path)
            changed += 1
            print(f"OK: {json_path}")
        except Exception as e:
            failed += 1
            print(f"BŁĄD: {json_path} -> {e}")

    print()
    print("Gotowe.")
    print(f"Przetworzono poprawnie: {changed}")
    print(f"Błędy: {failed}")


if __name__ == "__main__":
    main()
