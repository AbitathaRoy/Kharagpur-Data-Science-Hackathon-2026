# clean_cache.py
import json
import os

CACHE_FILE = "constraint_cache.json"


def main():
    if not os.path.exists(CACHE_FILE):
        print("No cache file found.")
        return

    with open(CACHE_FILE, 'r', encoding="utf-8") as f:
        cache = json.load(f)

    initial_count = len(cache)
    # Filter out empty lists (potentially failed chunks)
    # We keep only chunks that actually yielded constraints
    cleaned_cache = {k: v for k, v in cache.items() if v}

    final_count = len(cleaned_cache)
    removed = initial_count - final_count

    with open(CACHE_FILE, 'w', encoding="utf-8") as f:
        json.dump(cleaned_cache, f, indent=2)

    print(f"✅ Cache cleaned.")
    print(f"Removed {removed} empty entries (potential rate-limit failures).")
    print(f"Remaining valid entries: {final_count}")


if __name__ == "__main__":
    main()