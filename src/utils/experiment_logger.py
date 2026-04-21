from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExperimentLogger:
    csv_path: Path

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, row: dict[str, object]) -> None:
        fieldnames = list(row.keys())

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_fieldnames != fieldnames:
            merged_fieldnames = list(existing_fieldnames)
            for fieldname in fieldnames:
                if fieldname not in merged_fieldnames:
                    merged_fieldnames.append(fieldname)

            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=merged_fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
                writer.writerow(row)
            return

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(row)

    def next_experiment_id(self) -> int:
        if not self.csv_path.exists():
            return 1

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            max_id = 0
            for row in reader:
                raw_value = row.get("experiment_id", "")
                try:
                    experiment_id = int(raw_value)
                except (TypeError, ValueError):
                    continue

                if experiment_id > max_id:
                    max_id = experiment_id

        return max_id + 1

    @staticmethod
    def utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
