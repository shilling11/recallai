"""Second-stage gate + background perceiver worker + SQLite store for frame descriptions."""
import io
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import perceiver

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "recall.db"

HAMMING_MIN = 12       # of 64 bits; below this a frame counts as a duplicate of the last described one
STALE_S = 45           # pending frames older than this when a worker picks them up are dropped
MAX_SKIP_S = 5         # describe at least one frame this often, so memory never goes blind
WORKERS = 6

stats = {"received": 0, "described": 0, "skipped": 0, "dropped": 0, "errors": 0}

_pool = ThreadPoolExecutor(max_workers=WORKERS)
_lock = threading.Lock()
_last_hash = None
_last_described_t = 0
_newest_t = 0
_inflight = 0     # jobs running or about to run on the pool
_slot = None      # (t_ms, path) of the newest frame waiting for a free worker


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute(
        "CREATE TABLE IF NOT EXISTS frames "
        "(t_ms INTEGER PRIMARY KEY, path TEXT, status TEXT, desc TEXT, error TEXT, thumb INTEGER DEFAULT 0)"
    )
    return con


def freshness() -> dict:
    """How far behind real time the memory is: frames still waiting for the model, and age of the last description."""
    with _db() as con:
        pending = con.execute("SELECT COUNT(*) FROM frames WHERE status='pending'").fetchone()[0]
        last = con.execute("SELECT MAX(t_ms) FROM frames WHERE status IN ('described','skipped')").fetchone()[0]
    return {"pending": pending, "behind_ms": int(time.time() * 1000 - last) if last else None}


def dhash(data: bytes) -> int:
    img = Image.open(io.BytesIO(data)).convert("L").resize((9, 8))
    px = list(img.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def _drop(t_ms: int, path: Path):
    """A frame the model will never see: record it and free its file."""
    stats["dropped"] += 1
    with _db() as con:
        con.execute("UPDATE frames SET status='dropped', path=NULL WHERE t_ms=?", (t_ms,))
    path.unlink(missing_ok=True)


def _describe_job(t_ms: int, path: Path):
    if _newest_t - t_ms > STALE_S * 1000:
        _drop(t_ms, path)
        return
    try:
        desc = perceiver.describe(path)
        status, payload, err = "described", json.dumps(desc), None
        stats["described"] += 1
    except Exception as e:  # keep the worker alive, record the failure
        status, payload, err = "error", None, str(e)[:300]
        stats["errors"] += 1
    with _db() as con:
        con.execute("UPDATE frames SET status=?, desc=?, error=? WHERE t_ms=?", (status, payload, err, t_ms))


def _run(t_ms: int, path: Path):
    """Worker loop: describe a frame, then take the newest waiting one, until nothing is waiting."""
    global _inflight, _slot
    while True:
        _describe_job(t_ms, path)
        with _lock:
            if _slot is None:
                _inflight -= 1
                return
            (t_ms, path), _slot = _slot, None


def submit(t_ms: int, data: bytes, path: Path):
    """Called for every keyframe the phone sends. Decides whether the model sees it, and whether it is kept on disk.
    Duplicates are never written: they can't be evidence, and the model won't look at them."""
    global _last_hash, _last_described_t, _newest_t, _inflight, _slot
    stats["received"] += 1
    _newest_t = max(_newest_t, t_ms)
    h = dhash(data)
    replaced, dispatch = None, False
    # The file and the DB row are written inside the lock, before any worker can pick the frame up.
    with _lock:
        dup = (
            _last_hash is not None
            and bin(h ^ _last_hash).count("1") < HAMMING_MIN
            and (t_ms - _last_described_t) / 1000 < MAX_SKIP_S
        )
        if dup:
            with _db() as con:
                con.execute("INSERT OR REPLACE INTO frames (t_ms, path, status) VALUES (?,?,?)", (t_ms, None, "skipped"))
        else:
            _last_hash, _last_described_t = h, t_ms
            path.write_bytes(data)
            with _db() as con:
                con.execute("INSERT OR REPLACE INTO frames (t_ms, path, status) VALUES (?,?,?)", (t_ms, str(path), "pending"))
            if _inflight >= WORKERS:
                # All workers busy: keep only the NEWEST waiting frame (the settled scene matters most).
                replaced, _slot = _slot, (t_ms, path)
            else:
                _inflight += 1
                dispatch = True
    if dup:
        stats["skipped"] += 1
        return
    if replaced:
        _drop(*replaced)
    if dispatch:
        _pool.submit(_run, t_ms, path)
