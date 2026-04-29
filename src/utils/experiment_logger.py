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
        columns = list(row.keys())

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as file_handle:
                writer = csv.DictWriter(file_handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(row)
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            saved_columns = reader.fieldnames or []
            saved_rows = list(reader)

        if saved_columns != columns:
            merged_columns = list(saved_columns)
            for column in columns:
                if column not in merged_columns:
                    merged_columns.append(column)

            with self.csv_path.open("w", newline="", encoding="utf-8") as file_handle:
                writer = csv.DictWriter(file_handle, fieldnames=merged_columns)
                writer.writeheader()
                for saved_row in saved_rows:
                    writer.writerow(saved_row)
                writer.writerow(row)
            return

        with self.csv_path.open("a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=columns)
            writer.writerow(row)

    def next_experiment_id(self) -> int:
        if not self.csv_path.exists():
            return 1

        with self.csv_path.open("r", newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            highest_id = 0
            for row in reader:
                raw_value = row.get("experiment_id", "")
                try:
                    experiment_id = int(raw_value)
                except (TypeError, ValueError):
                    continue

                if experiment_id > highest_id:
                    highest_id = experiment_id

        return highest_id + 1

    @staticmethod
    def utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
