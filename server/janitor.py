"""Storage janitor: keep full-quality frames only where they prove an event; shrink the rest to thumbnails."""
import os
import threading
import time
from pathlib import Path

from PIL import Image

from . import episodes, memory
from .pipeline import _db

AGE_S = 180           # a frame must be this old before it can be shrunk
THUMB_WIDTH = 320
THUMB_QUALITY = 60
INTERVAL_S = 30


def shrink_old_frames() -> int:
    """Thumbnail every described frame older than AGE_S that is not the evidence frame of an event."""
    memory.rebuild_events()
    cutoff = int((time.time() - AGE_S) * 1000)
    with _db() as con:
        evidence = {r[0] for r in con.execute("SELECT t_ms FROM events")}
    evidence |= {e["key_frame_ms"] for e in episodes.build() if e["kind"] == "episode"}   # keep each episode's picture sharp
    evidence |= {p["key_frame_ms"] for p in episodes.build_people()}
    evidence |= {v["key_frame_ms"] for v in episodes.build_visits()} | {t["key_frame_ms"] for t in episodes.build_texts()}
    with _db() as con:
        rows = con.execute(
            "SELECT t_ms, path FROM frames WHERE status IN ('described','error') AND thumb=0 AND path IS NOT NULL AND t_ms<?",
            (cutoff,),
        ).fetchall()
    done = []
    for t_ms, path in rows:
        if t_ms in evidence or not os.path.exists(path):
            continue
        img = Image.open(path)
        img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
        img.save(path, "JPEG", quality=THUMB_QUALITY)
        done.append(t_ms)
    if done:
        with _db() as con:
            con.executemany("UPDATE frames SET thumb=1 WHERE t_ms=?", [(t,) for t in done])
    return len(done)


def storage_stats() -> dict:
    """Numbers for the pitch: how much was kept versus how much the gates avoided storing."""
    with _db() as con:
        rows = con.execute("SELECT status, thumb, path FROM frames").fetchall()
    out = {"seen_by_server": len(rows), "not_stored": 0, "full_frames": 0, "thumbnails": 0, "disk_mb": 0.0}
    total = 0
    for status, thumb, path in rows:
        if not path:
            out["not_stored"] += 1
            continue
        out["thumbnails" if thumb else "full_frames"] += 1
        if os.path.exists(path):
            total += os.path.getsize(path)
    out["disk_mb"] = round(total / 1e6, 2)
    return out


def start():
    def loop():
        while True:
            time.sleep(INTERVAL_S)
            try:
                shrink_old_frames()
            except Exception as e:  # never let housekeeping take the server down
                print("janitor error:", e)
    threading.Thread(target=loop, daemon=True).start()
