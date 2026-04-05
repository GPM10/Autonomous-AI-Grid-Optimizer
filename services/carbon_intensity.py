"""Utility helpers for the National Grid Carbon Intensity API."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_BASE = "https://api.carbonintensity.org.uk"
USER_AGENT = "autonomous-grid-optimizer/1.0"


def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _request(path: str) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_national_intensity(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch national carbon intensity between two datetimes."""
    payload = _request(f"/intensity/{_format_iso(start)}/{_format_iso(end)}")
    return payload.get("data", [])


def fetch_regional_intensity(
    start: datetime,
    end: datetime,
    region_id: Optional[int] = None,
    postcode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch regional carbon intensity between two datetimes."""
    path = f"/regional/intensity/{_format_iso(start)}/{_format_iso(end)}"
    if region_id is not None:
        path += f"/regionid/{region_id}"
    elif postcode:
        path += f"/postcode/{quote(postcode)}"
    payload = _request(path)
    return payload.get("data", [])


def default_window(hours: int = 48) -> Dict[str, datetime]:
    """Return a UTC-aligned default start/end window."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return {"start": now, "end": now + timedelta(hours=hours)}
