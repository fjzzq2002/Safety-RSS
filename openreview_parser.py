#!/usr/bin/env python3
"""
Fetch submissions from the OpenReview API and export them as newline-delimited JSON.

Usage:
    python openreview_parser.py --url "https://api2.openreview.net/notes?content.venueid=ICLR.cc%2F2026%2FConference%2FSubmission&details=replyCount%2Cpresentation%2Cwritable&domain=ICLR.cc%2F2026%2FConference&limit=1000&offset=0" --output iclr2026_preprints.jsonl

The script automatically paginates by incrementing the `offset` query parameter until
the API returns an empty list of notes.  Each note is normalised into a compact record
containing the title, abstract, authors, link, timestamp, and other useful metadata
that can be consumed by the RSS tooling (e.g. the JSONL loader in main.py).

Network access can be disabled by supplying `--input-file example.json`, which should
contain a previously-downloaded OpenReview JSON response (with a top-level `notes`
array).  This is useful for testing or regenerating frozen feeds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

try:
    from fake_useragent import UserAgent
except Exception:  # pragma: no cover - fallback if library missing
    UserAgent = None  # type: ignore


DEFAULT_BATCH_SIZE = 1000
DEFAULT_TIMEOUT = 30
_USER_AGENT: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OpenReview notes and output JSONL.")
    parser.add_argument(
        "--url",
        required=False,
        help="Base OpenReview API URL (e.g. https://api2.openreview.net/notes?...).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("openreview_notes.jsonl"),
        help="Path to write newline-delimited JSON (default: openreview_notes.jsonl).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of notes per request (overrides URL limit parameter).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional safety cap on the number of pages to fetch.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional path to a cached OpenReview response (skips network requests).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the first few records to stdout for inspection.",
    )
    args = parser.parse_args()
    if not args.url and not args.input_file:
        parser.error("Either --url or --input-file must be supplied.")
    return args


def update_query(url: str, offset: int, limit: int) -> str:
    parsed = urlparse(url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params["offset"] = str(offset)
    query_params["limit"] = str(limit)
    updated_query = urlencode(query_params, doseq=False)
    return urlunparse(parsed._replace(query=updated_query))


def get_user_agent() -> Optional[str]:
    global _USER_AGENT
    if _USER_AGENT is not None:
        return _USER_AGENT
    if UserAgent is None:
        return None
    try:
        _USER_AGENT = UserAgent().random.strip()
    except Exception:
        _USER_AGENT = None
    return _USER_AGENT


def fetch_notes_from_api(
    url: str,
    batch_size: int,
    max_pages: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    session = requests.Session()
    ua = get_user_agent()
    if ua:
        session.headers.update({"User-Agent": ua})
    page = 0
    offset = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        paged_url = update_query(url, offset=offset, limit=batch_size)
        sys.stderr.write(f"Fetching {paged_url}\n")
        response = session.get(paged_url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        notes = payload.get("notes", [])
        if not notes:
            break
        for note in notes:
            yield note
        page += 1
        offset += batch_size


def iter_notes(
    url: Optional[str],
    batch_size: int,
    max_pages: Optional[int],
    input_file: Optional[Path],
) -> Iterable[Dict[str, Any]]:
    if input_file:
        sys.stderr.write(f"Loading cached notes from {input_file}\n")
        data = json.loads(input_file.read_text(encoding="utf-8"))
        notes = data.get("notes", [])
        sys.stderr.write(f"Loaded {len(notes)} notes from cache\n")
        return notes
    if not url:
        return []
    return fetch_notes_from_api(url, batch_size=batch_size, max_pages=max_pages)


def to_iso_timestamp(ts_ms: Optional[int]) -> Optional[str]:
    if not ts_ms:
        return None
    try:
        return dt.datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"
    except Exception:
        return None


def normalise_note(note: Dict[str, Any]) -> Dict[str, Any]:
    content = note.get("content") or {}

    def extract_field(field: str) -> Optional[Any]:
        value = content.get(field)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    title = extract_field("title") or note.get("forum") or "Untitled"
    abstract = extract_field("abstract") or ""
    keywords = extract_field("keywords") or []
    if isinstance(keywords, dict) and "value" in keywords:
        keywords = keywords["value"]

    if isinstance(keywords, list):
        keywords = [kw for kw in keywords if isinstance(kw, str)]
    else:
        keywords = []

    raw_authors = extract_field("authors") or extract_field("authorids") or []
    authors: List[str] = []
    if isinstance(raw_authors, list):
        authors = [str(a).strip() for a in raw_authors if str(a).strip()]
    elif isinstance(raw_authors, str):
        authors = [part.strip() for part in raw_authors.split(";") if part.strip()]

    pdf_path = extract_field("pdf")
    if isinstance(pdf_path, str) and pdf_path and not pdf_path.startswith("http"):
        pdf_url = f"https://openreview.net{pdf_path}"
    elif isinstance(pdf_path, str):
        pdf_url = pdf_path
    else:
        pdf_url = None

    forum_id = note.get("forum") or note.get("id")
    link = f"https://openreview.net/forum?id={forum_id}" if forum_id else None

    record: Dict[str, Any] = {
        "id": note.get("id"),
        "forum": forum_id,
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "authors": authors,
        "venue": extract_field("venue"),
        "venueid": extract_field("venueid"),
        "primary_area": extract_field("primary_area"),
        "link": link,
        "pdf_url": pdf_url,
        "pubDate": to_iso_timestamp(note.get("cdate")),
        "updated": to_iso_timestamp(note.get("mdate")),
        "cdate": to_iso_timestamp(note.get("cdate")),
        "tcdate": to_iso_timestamp(note.get("tcdate")),
        "mdate": to_iso_timestamp(note.get("mdate")),
        "details": note.get("details"),
    }
    return record


def main() -> None:
    args = parse_args()
    notes_iterable = iter_notes(
        url=args.url,
        batch_size=args.batch_size,
        max_pages=args.max_pages,
        input_file=args.input_file,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    preview_records: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as fh:
        for raw_note in notes_iterable:
            record = normalise_note(raw_note)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if args.pretty and len(preview_records) < 3:
                preview_records.append(record)

    sys.stderr.write(f"Wrote {count} notes to {output_path}\n")

    if args.pretty and preview_records:
        json.dump(preview_records, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted by user.\n")
