import ast
import csv
import os

_CSV_PATH = os.path.join(os.path.dirname(__file__), "described_dataset.csv")


def _load_database() -> list[dict]:
    assets = []
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                size = ast.literal_eval(row["size"])
            except Exception:
                size = [1.0, 1.0, 1.0]
            if row["suggested_db_id"] == "unknown_object":
                continue
            assets.append({
                "db_id":       row["suggested_db_id"],
                "name":        row["original_name"],
                "description": row["description"],
                "size":        size,
            })
    return assets


ASSET_DATABASE = _load_database()
