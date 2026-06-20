"""Data loading for the WC2022 channel-pass study (event data only).

Two on-disk layouts are supported (auto-detected):

* **Subdir layout** (the full PFF release):
    {data_dir}/Metadata/{match_id}.json
    {data_dir}/Event Data/{match_id}.json
* **Flat layout** (single-match example):
    {data_dir}/{match_id}_meta.json
    {data_dir}/{match_id}_event.json

PFF datasets used here
----------------------
* Metadata : team names/colours, pitch size.
* Event    : one row per on-ball event. Carries ``linesBrokenType`` (which
             opponent lines a pass broke: D/M/A) and a positional snapshot of
             all 22 players with ``positionGroupType`` — enough to detect the
             defensive line and the receiver's position without tracking data.

Coordinate convention (centre-origin metres)
  x : -52.5 ... +52.5  (goal to goal)
  y : -34.0 ... +34.0  (touchline to touchline)
"""
from __future__ import annotations

import json
from pathlib import Path


def _resolve(data_dir: str | Path, match_id: int | str, kind: str) -> Path:
    """Path to one match file, trying the subdir then flat layout."""
    d = Path(data_dir)
    candidates = {
        "meta":  [d / "Metadata" / f"{match_id}.json",   d / f"{match_id}_meta.json"],
        "event": [d / "Event Data" / f"{match_id}.json",  d / f"{match_id}_event.json"],
    }[kind]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default to subdir path for clearer error messages


def available_matches_event_only(data_dir: str | Path = "data") -> list[str]:
    """Return match ids (sorted) that have both metadata + event data.

    Channel-pass extraction is event-only, so this returns all 64 WC2022 matches
    (not just the ~50 that also ship tracking data).
    """
    d = Path(data_dir)

    def _ids(folder: Path, suffix: str, flat_suffix: str) -> set[str]:
        ids: set[str] = set()
        if folder.exists():
            ids |= {p.name[: -len(suffix)] for p in folder.glob(f"*{suffix}")}
        ids |= {p.name[: -len(flat_suffix)] for p in d.glob(f"*{flat_suffix}")}
        return ids

    meta  = _ids(d / "Metadata", ".json", "_meta.json")
    event = _ids(d / "Event Data", ".json", "_event.json")
    return sorted((i for i in (meta & event) if i.isdigit()), key=int)


def load_meta(match_id: int | str, data_dir: str | Path = "data") -> dict:
    """Load match metadata for *match_id* (either layout)."""
    with open(_resolve(data_dir, match_id, "meta"), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw[0] if isinstance(raw, list) else raw


def load_events(match_id: int | str, data_dir: str | Path = "data") -> list[dict]:
    """Load event data (list of event dicts) for *match_id* (either layout)."""
    with open(_resolve(data_dir, match_id, "event"), "r", encoding="utf-8") as fh:
        return json.load(fh)
