from __future__ import annotations

import csv
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


META_COLUMNS = ["person_id", "name", "image_path", "created_at"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_name(name: str) -> str:
    return " ".join((name or "").strip().split())


def person_id_from_name(name: str) -> str:
    normalized = normalize_name(name).lower()
    if not normalized:
        raise ValueError("Empty name")
    # Deterministic ID so the same name maps to the same person_id across runs.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"facerec:{normalized}"))


def embedding_columns(embedding_dim: int) -> list[str]:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be > 0")
    return [f"embedding_{i}" for i in range(embedding_dim)]


def expected_header(embedding_dim: int) -> list[str]:
    return [*META_COLUMNS, *embedding_columns(embedding_dim)]


@dataclass
class UsersSummary:
    person_id: str
    name: str
    samples: int


class EmbeddingsStore:
    def __init__(self, csv_path: Path, embedding_dim: int):
        self.csv_path = Path(csv_path)
        self.embedding_dim = int(embedding_dim)
        self.header = expected_header(self.embedding_dim)
        self._seen_image_paths: set[str] = set()
        self._lock = threading.RLock()

    def ensure_exists(self) -> None:
        with self._lock:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if self.csv_path.exists():
                return
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.header)
                writer.writeheader()

    def load_index(self) -> None:
        with self._lock:
            self._seen_image_paths.clear()
            if not self.csv_path.exists():
                return

            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return
                missing = [c for c in META_COLUMNS if c not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Embeddings CSV missing columns: {missing}")

                for row in reader:
                    image_path = (row.get("image_path") or "").strip()
                    if image_path:
                        self._seen_image_paths.add(image_path)

    def already_processed(self, image_path: str) -> bool:
        with self._lock:
            return (image_path or "").strip() in self._seen_image_paths

    def append_rows(self, rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        with self._lock:
            self.ensure_exists()

            appended = 0
            skipped = 0

            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.header)

                for row in rows:
                    image_path = str(row.get("image_path", "")).strip()
                    if image_path and (image_path in self._seen_image_paths):
                        skipped += 1
                        continue

                    out: dict[str, Any] = {k: row.get(k, "") for k in self.header}
                    # Ensure all embedding columns exist.
                    for c in embedding_columns(self.embedding_dim):
                        if c not in out:
                            out[c] = ""

                    writer.writerow(out)
                    if image_path:
                        self._seen_image_paths.add(image_path)
                    appended += 1

            return {"appended": appended, "skipped": skipped}

    def users(self) -> list[UsersSummary]:
        if not self.csv_path.exists():
            return []

        counts: dict[tuple[str, str], int] = {}
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                person_id = (row.get("person_id") or "").strip()
                name = (row.get("name") or "").strip()
                if not person_id or not name:
                    continue
                key = (person_id, name)
                counts[key] = counts.get(key, 0) + 1

        summaries = [UsersSummary(person_id=k[0], name=k[1], samples=v) for k, v in counts.items()]
        summaries.sort(key=lambda s: (s.name.lower(), s.person_id))
        return summaries

    def total_rows(self) -> int:
        if not self.csv_path.exists():
            return 0
        total = 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for _ in reader:
                total += 1
        return total
