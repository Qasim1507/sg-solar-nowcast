"""Shared HTTP client, config loader and manifest helpers for all fetchers.

Engineering requirements this file exists to satisfy:
  * resumable   - on-disk state, re-running skips completed work
  * idempotent  - writes are atomic (tmp + rename), re-fetch overwrites cleanly
  * rate-limited- polite delay between requests, honours server backoff hints
  * retrying    - urllib3.Retry with exponential backoff on transient failures

The data.gov.sg quirk that motivates `json_code_field`:
    A rate-limited response comes back as **HTTP 200** with `{"code": 24,
    "errorMsg": "Rate limit exceeded..."}` in the body. `raise_for_status()`
    sails straight past it. Verified 2026-09-08.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger("fetch")


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


class RateLimitError(RuntimeError):
    """Server told us to slow down (may be signalled in-body, not by status)."""


class PermanentError(RuntimeError):
    """A 4xx that will never succeed on retry - e.g. 404 for a day with no data.

    Retrying these is not merely useless, it is actively harmful: six attempts with
    exponential backoff burns ~60s per bad day and can dominate a long backfill.
    """


class ApiClient:
    """requests.Session with retry/backoff plus in-body error-code inspection."""

    def __init__(
        self,
        delay_s: float = 0.0,
        backoff_s: float = 11.0,
        total_retries: int = 5,
        timeout: int = 90,
        json_code_field: str | None = None,
        user_agent: str = "sg-solar-nowcast/0.1",
    ):
        self.delay_s = delay_s
        self.backoff_s = backoff_s
        self.timeout = timeout
        self.json_code_field = json_code_field
        self._last_request = 0.0

        retry = Retry(
            total=total_retries,
            connect=total_retries,
            read=total_retries,
            status=total_retries,
            backoff_factor=1.5,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": user_agent})

    def _throttle(self) -> None:
        if self.delay_s <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_request = time.monotonic()

    def get_json(self, url: str, params: dict | None = None, max_attempts: int = 6) -> dict:
        """GET returning parsed JSON, retrying through transport AND in-body errors."""
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    raise RateLimitError("HTTP 429")
                # 4xx other than 429 will not become 2xx by waiting.
                if 400 <= r.status_code < 500:
                    raise PermanentError(f"HTTP {r.status_code} for {r.url[:120]}")
                r.raise_for_status()
                payload = r.json()

                # In-body error code (data.gov.sg): HTTP 200 but code != 0.
                if self.json_code_field is not None:
                    code = payload.get(self.json_code_field)
                    if code not in (0, None):
                        msg = str(payload.get("errorMsg", ""))
                        if code == 24 or "rate limit" in msg.lower():
                            raise RateLimitError(f"code={code}: {msg[:80]}")
                        raise RuntimeError(f"API error code={code}: {msg[:120]}")

                # Open-Meteo signals errors with {"error": true, "reason": ...}
                if payload.get("error"):
                    raise RuntimeError(f"API error: {str(payload.get('reason'))[:160]}")
                return payload

            except PermanentError:
                raise
            except RateLimitError as e:
                last_err = e
                wait = self.backoff_s * (1 + attempt * 0.5)
                log.warning("rate limited (%s); sleeping %.0fs", e, wait)
                time.sleep(wait)
            except (requests.RequestException, ValueError) as e:
                last_err = e
                wait = 2.0 * (attempt + 1)
                log.warning("request failed (%s); retry in %.0fs", str(e)[:100], wait)
                time.sleep(wait)

        raise RuntimeError(f"GET failed after {max_attempts} attempts: {url} :: {last_err}")


# ------------------------------------------------------------------ atomic IO

def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class ResumeState:
    """Tracks completed work units so a re-run skips them."""

    def __init__(self, path: Path):
        self.path = path
        self.done: set[str] = set(read_json(path) or [])

    def has(self, key: str) -> bool:
        return key in self.done

    def add(self, key: str, flush: bool = False) -> None:
        self.done.add(key)
        if flush:
            self.flush()

    def flush(self) -> None:
        write_json_atomic(self.path, sorted(self.done))
