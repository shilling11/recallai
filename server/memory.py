"""Memory layer: canonical object names, state-change events, and the query functions the agent calls."""
import json
import re
import sqlite3

from .pipeline import DB_PATH

# The VLM names the same thing differently frame to frame; fold the variants together.
ALIASES = {
    "ring": "keys", "key": "keys", "key ring": "keys", "keyring": "keys", "keychain": "keys", "key chain": "keys",
    "car keys": "keys", "house keys": "keys", "set of keys": "keys", "keys on ring": "keys",
    "cell phone": "phone", "smartphone": "phone", "mobile phone": "phone", "iphone": "phone",
    "table": "desk", "work desk": "desk", "workspace": "desk",   # same surface as far as the demo is concerned
    "juice carton": "carton", "juice box": "carton", "juice": "carton", "box": "carton", "drink carton": "carton",
    "cup": "mug", "coffee mug": "mug", "coffee cup": "mug",
    "notebook computer": "laptop", "macbook": "laptop", "computer": "laptop",
    "glasses": "glasses", "spectacles": "glasses", "sunglasses": "glasses",
}
TRACKED_STATES = {"held": "held", "on_surface": "placed", "in_container": "placed", "worn": "worn"}
CONFIRM = 2  # a new state must be seen in this many consecutive described frames to count


# Keyword rules catch the long tail of names the dict can't enumerate ("earbud case", "airpods pro", ...).
KEYWORD_RULES = [
    (r"\b(earbuds?|earphones?|airpods?|earpods?|headphones?|headset)\b", "earbuds"),
    (r"\bkeys?\b|\bkey ?(ring|chain)\b", "keys"),
    (r"\b(carton|juice|drink box)\b", "carton"),
]


def canon(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().lower())
    if n in ALIASES:
        return ALIASES[n]
    for pattern, target in KEYWORD_RULES:
        if re.search(pattern, n):
            return target
    return n


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, t_ms INTEGER, obj TEXT, kind TEXT, place TEXT, summary TEXT)"
    )
    return con


def _place(frame: dict, obj: str) -> str:
    """Best guess for where an object is: its 'on/in' relation target, else the frame's location."""
    for r in frame.get("relations", []):
        if canon(r.get("subj", "")) == obj and r.get("pred") in ("on", "in", "near") and r.get("obj"):
            target = canon(r["obj"])
            if target != "wearer":
                return f'{r["pred"]} {target}'
    return f'at {frame.get("location") or "unknown"}'


def _observations():
    """Per described frame (time order): {obj: (state, place)} plus the frame."""
    with _con() as con:
        rows = con.execute("SELECT t_ms, desc FROM frames WHERE status='described' ORDER BY t_ms").fetchall()
    for r in rows:
        f = json.loads(r["desc"])
        seen = {}
        for o in f.get("objects", []):
            state = TRACKED_STATES.get(o.get("state"))
            if state:
                name = canon(o["name"])
                seen[name] = (state, _place(f, name))
        yield r["t_ms"], seen, f


def rebuild_events() -> int:
    """Recompute events from scratch: a debounced state change per object becomes one event."""
    current, pending = {}, {}   # obj -> confirmed state ; obj -> (candidate, count, t_first, place)
    events = []
    for t_ms, seen, _ in _observations():
        for obj, (state, place) in seen.items():
            if current.get(obj, (None,))[0] == state:
                pending.pop(obj, None)
                current[obj] = (state, place)   # keep place fresh while the state holds
                continue
            cand, count, t_first, _p = pending.get(obj, (state, 0, t_ms, place))
            if cand != state:
                cand, count, t_first = state, 0, t_ms
            count += 1
            pending[obj] = (cand, count, t_first, place)
            if count >= CONFIRM:
                prev = current.get(obj, (None,))[0]
                kind = {"held": "picked_up", "placed": "put_down", "worn": "put_on"}[state]
                if prev is None:
                    kind = {"held": "seen_held", "placed": "seen_placed", "worn": "seen_worn"}[state]
                events.append((t_first, obj, kind, place))
                current[obj] = (state, place)
                pending.pop(obj, None)
    with _con() as con:
        con.execute("DELETE FROM events")
        con.executemany(
            "INSERT INTO events (t_ms, obj, kind, place, summary) VALUES (?,?,?,?,?)",
            [(t, o, k, p, f"{o}: {k.replace('_', ' ')} {p}") for t, o, k, p in events],
        )
    return len(events)


# ---- scene graph ----

WINDOW_MS = 6000   # frames this close together are merged into one snapshot of the scene
REL_PREDS = {"on", "in", "near", "holding", "picked_up", "put_down"}


def graph_at(t_ms: int | None = None, with_markers: bool = True) -> dict:
    """Scene graph as plain JSON: nodes (wearer, surfaces, objects) and labelled edges, as of t_ms (default: latest).
    Merges the described frames in the preceding WINDOW_MS; an edge needs to appear in 2+ frames when 2+ exist,
    which filters one-frame hallucinations."""
    with _con() as con:
        bounds = con.execute("SELECT MIN(t_ms), MAX(t_ms) FROM frames WHERE status='described'").fetchone()
        if bounds[0] is None:
            return {"t_ms": None, "t_min": None, "t_max": None, "nodes": [], "edges": []}
        t = min(t_ms or bounds[1], bounds[1])
        rows = con.execute(
            "SELECT t_ms, desc FROM frames WHERE status='described' AND t_ms<=? AND t_ms>? ORDER BY t_ms",
            (t, t - WINDOW_MS),
        ).fetchall()
    n = len(rows)
    need = 2 if n >= 2 else 1
    edge_count, node_state, node_count = {}, {}, {}
    for r in rows:
        f = json.loads(r["desc"])
        seen_edges, seen_nodes = set(), {}
        for o in f.get("objects", []):
            seen_nodes[canon(o["name"])] = o.get("state", "unknown")
        for rel in f.get("relations", []):
            sj, pr, ob = canon(rel.get("subj", "")), rel.get("pred"), canon(rel.get("obj", ""))
            if sj and ob and pr in REL_PREDS and sj != ob:
                seen_edges.add((sj, pr, ob))
        for e in seen_edges:
            edge_count[e] = edge_count.get(e, 0) + 1
        for name, state in seen_nodes.items():
            node_count[name] = node_count.get(name, 0) + 1
            node_state[name] = state            # latest state wins
    edges = [{"s": s_, "p": p_, "o": o_, "n": c} for (s_, p_, o_), c in edge_count.items() if c >= need]
    names = {e["s"] for e in edges} | {e["o"] for e in edges} | {k for k, c in node_count.items() if c >= need}
    # A held or worn object is attached to the wearer even if the model gave no explicit relation for it.
    linked = {(e["s"], e["o"]) for e in edges}
    for name in sorted(names):
        st = node_state.get(name)
        if name != "wearer" and st in ("held", "worn") and ("wearer", name) not in linked:
            edges.append({"s": "wearer", "p": "holding" if st == "held" else "wearing", "o": name, "n": node_count.get(name, 1)})
    if any(e["s"] == "wearer" or e["o"] == "wearer" for e in edges):
        names.add("wearer")
    surfaces = {e["o"] for e in edges if e["p"] in ("on", "in")} - {e["s"] for e in edges if e["p"] in ("on", "in")}
    nodes = []
    for name in sorted(names):
        kind = "person" if name == "wearer" else "surface" if name in surfaces and name not in node_state else "object"
        nodes.append({"id": name, "kind": kind, "state": node_state.get(name)})
    with _con() as con:
        fr = con.execute(
            "SELECT t_ms FROM frames WHERE status='described' AND path IS NOT NULL AND t_ms<=? ORDER BY t_ms DESC LIMIT 1", (t,)
        ).fetchone()
        marks = [dict(r) for r in con.execute(
            "SELECT t_ms, obj, kind FROM events WHERE kind IN ('picked_up','put_down','put_on') ORDER BY t_ms LIMIT 200")] if with_markers else []
    return {"t_ms": t, "t_min": bounds[0], "t_max": bounds[1], "nodes": nodes, "edges": edges,
            "frame_t_ms": fr["t_ms"] if fr else None, "markers": marks}


def graph_series(n: int = 120) -> dict:
    """Snapshots of the scene graph across the whole session, for replay and scrubbing with no further requests.
    Times are spread evenly, plus one snapshot at every event marker so jumping to an event is exact."""
    first = graph_at(None, with_markers=True)
    if first["t_ms"] is None:
        return {"t_min": None, "t_max": None, "markers": [], "steps": []}
    lo, hi = first["t_min"], first["t_max"]
    n = max(2, min(n, 240))
    times = {lo + round((hi - lo) * i / (n - 1)) for i in range(n)} | {m["t_ms"] for m in first["markers"] if lo <= m["t_ms"] <= hi}
    steps = []
    for t in sorted(times):
        g = graph_at(t, with_markers=False)
        steps.append({"t_ms": g["t_ms"], "frame_t_ms": g["frame_t_ms"], "nodes": g["nodes"], "edges": g["edges"]})
    return {"t_min": lo, "t_max": hi, "markers": first["markers"], "steps": steps}


# ---- queries used by the agent ----

def last_seen(obj: str) -> dict | None:
    """Most recent confirmed event for an object, plus the frame to prove it."""
    obj = canon(obj)
    with _con() as con:
        e = con.execute("SELECT * FROM events WHERE obj=? ORDER BY t_ms DESC LIMIT 1", (obj,)).fetchone()
        frame = None
        if e:
            frame = con.execute(
                "SELECT t_ms, path FROM frames WHERE status='described' AND path IS NOT NULL ORDER BY ABS(t_ms-?) LIMIT 1", (e["t_ms"],)
            ).fetchone()
    sighting = latest_sighting(obj, after_ms=e["t_ms"] if e else 0)
    if sighting and e:
        same_state = {"picked_up": "held", "put_down": "placed", "put_on": "worn"}.get(e["kind"], "placed") == sighting["state"]
        if same_state and sighting["place"] == e["place"]:
            sighting = None   # just reinforces the confirmed event, nothing new to say
    if not e and not sighting:
        return None
    out = {**dict(e), "frame_t_ms": frame["t_ms"] if frame else None} if e else {}
    if sighting:
        out["latest_sighting"] = sighting
    return out


def latest_sighting(obj: str, after_ms: int = 0, window_ms: int = 60000) -> dict | None:
    """Most recent single-frame observation of obj that is newer than the last confirmed event.
    Unconfirmed (one frame), so the agent must hedge. Only the last minute counts."""
    obj = canon(obj)
    with _con() as con:
        rows = con.execute(
            "SELECT t_ms, desc FROM frames WHERE status='described' AND t_ms>? ORDER BY t_ms DESC LIMIT 20", (after_ms,)
        ).fetchall()
    newest = rows[0]["t_ms"] if rows else 0
    for r in rows:
        if newest - r["t_ms"] > window_ms:
            break
        f = json.loads(r["desc"])
        for o in f.get("objects", []):
            if canon(o["name"]) == obj and TRACKED_STATES.get(o.get("state")):
                return {"state": TRACKED_STATES[o["state"]], "place": _place(f, obj), "t_ms": r["t_ms"], "confirmed": False}
    return None


def events_for(obj: str | None = None, since_ms: int | None = None, limit: int = 50) -> list[dict]:
    q, args = "SELECT * FROM events WHERE 1=1", []
    if obj:
        q += " AND obj=?"; args.append(canon(obj))
    if since_ms:
        q += " AND t_ms>=?"; args.append(since_ms)
    q += " ORDER BY t_ms DESC LIMIT ?"; args.append(limit)
    with _con() as con:
        return [dict(r) for r in con.execute(q, args)][::-1]


if __name__ == "__main__":
    print(rebuild_events(), "events")
    for e in events_for():
        print(e["t_ms"] % 1000000, e["summary"])
