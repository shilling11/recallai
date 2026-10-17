"""Episodes: collapse per-frame descriptions into 'what I was doing, where, and for how long'.

Derived on demand from the described frames (nothing extra to store or keep in sync). Boundaries come from code,
not from the model: runs of the same (place, activity), short runs absorbed into neighbours, and long silences
recorded as gaps so the agent never claims "you did nothing" when the camera simply wasn't seeing.
"""
import json
import re

from .memory import TRACKED_STATES, _con, canon

ACTIVITIES = ["working", "walking", "eating_drinking", "using_phone", "talking", "resting", "other"]
MIN_EPISODE_S = 30   # a run shorter than this is absorbed into its neighbour (single-frame flicker)
GAP_S = 45           # no described frame for this long means the camera was off or blocked

OUTDOOR_WORDS = ("outside", "outdoor", "street", "sidewalk", "pavement", "patio", "garden", "road", "car park", "entrance")
PLACE_ALIASES = {"table": "desk", "office": "desk", "workspace": "desk", "work desk": "desk", "computer desk": "desk"}
_ACTIVITY_WORDS = [  # fallback for frames described before the model returned an activity label
    ("working", ("laptop", "typing", "computer", "screen", "desk", "working", "mouse")),
    ("using_phone", ("phone",)),
    ("eating_drinking", ("eating", "drinking", "mug", "juice", "carton", "coffee")),
    ("walking", ("walking", "standing", "moving")),
    ("talking", ("talking", "speaking", "conversation")),
    ("resting", ("resting", "sitting", "sleeping", "relaxing")),
]


def _norm_place(loc) -> str | None:
    p = re.sub(r"^(a|an|the)\s+", "", str(loc or "").strip().lower())
    p = re.sub(r"^(at|in|on)\s+", "", p)
    if p in ("", "unknown", "n/a", "none", "null"):
        return None
    return PLACE_ALIASES.get(p, p)


def _norm_activity(f: dict) -> str:
    a = str(f.get("activity", "")).strip().lower().replace(" ", "_")
    if a in ACTIVITIES:
        return a
    text = str(f.get("action", "")).lower()
    for label, words in _ACTIVITY_WORDS:
        if any(w in text for w in words):
            return label
    return "other"


def _load_frames() -> list[dict]:
    with _con() as con:
        rows = con.execute(
            "SELECT t_ms, path, desc FROM frames WHERE status='described' ORDER BY t_ms"
        ).fetchall()
    frames = []
    for r in rows:
        f = json.loads(r["desc"])
        objs = {}
        for o in f.get("objects", []):
            state = TRACKED_STATES.get(o.get("state"))
            if state:
                objs[canon(o["name"])] = state
        raw_text = f.get("visible_text") or []
        texts = [str(x).strip() for x in (raw_text if isinstance(raw_text, list) else [raw_text]) if str(x).strip()][:6]
        setting = str(f.get("setting", "")).strip().lower()
        loc = str(f.get("location", "")).lower()
        if setting not in ("indoor", "outdoor"):
            setting = "outdoor" if any(w in loc for w in OUTDOOR_WORDS) else None
        try:
            people = max(0, int(f.get("people_visible") or 0))
        except (TypeError, ValueError):
            people = 0
        frames.append({"t": r["t_ms"], "path": r["path"], "place": _norm_place(f.get("location")),
                       "activity": _norm_activity(f), "objects": objs, "people": people,
                       "setting": setting, "texts": texts})
    return frames


def segment(frames: list[dict]) -> list[dict]:
    """Pure function: described frames in time order -> list of episodes and gaps."""
    runs, last_place = [], None
    for fr in frames:
        place = fr["place"] or last_place          # an unreadable location inherits the previous one
        last_place = place or last_place
        gap = runs and runs[-1]["kind"] == "episode" and fr["t"] - runs[-1]["frames"][-1]["t"] > GAP_S * 1000
        if gap:
            runs.append({"kind": "gap", "frames": [runs[-1]["frames"][-1], fr]})
        if not runs or runs[-1]["kind"] == "gap" or (runs[-1]["place"], runs[-1]["activity"]) != (place, fr["activity"]):
            runs.append({"kind": "episode", "place": place, "activity": fr["activity"], "frames": []})
        runs[-1]["frames"].append(fr)

    def duration(run):
        return run["frames"][-1]["t"] - run["frames"][0]["t"]

    changed = True
    while changed and len(runs) > 1:                # absorb flicker into neighbours, then re-merge equal neighbours
        changed = False
        for i, run in enumerate(runs):
            if run["kind"] != "episode" or duration(run) >= MIN_EPISODE_S * 1000:
                continue
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs) and runs[j]["kind"] == "episode"]
            if not nbrs:
                continue
            j = nbrs[0]
            runs[j]["frames"] = sorted(runs[j]["frames"] + run["frames"], key=lambda f: f["t"])
            del runs[i]
            changed = True
            break
        k = 0
        while k < len(runs) - 1:
            a, b = runs[k], runs[k + 1]
            if a["kind"] == b["kind"] == "episode" and (a["place"], a["activity"]) == (b["place"], b["activity"]):
                a["frames"] += b["frames"]
                del runs[k + 1]
                changed = True
            else:
                k += 1

    out = []
    for i, run in enumerate(runs):
        fr = run["frames"]
        if run["kind"] == "gap":
            out.append({"kind": "gap", "start_ms": fr[0]["t"], "end_ms": fr[-1]["t"]})
            continue
        counts, held = {}, {}
        for f in fr:
            for name, state in f["objects"].items():
                counts[name] = counts.get(name, 0) + 1
                if state == "held":
                    held[name] = held.get(name, 0) + 1
        need = 2 if len(fr) >= 2 else 1
        mid = (fr[0]["t"] + fr[-1]["t"]) / 2
        with_file = [f for f in fr if f["path"]] or fr
        key = min(with_file, key=lambda f: abs(f["t"] - mid))
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        end = nxt["frames"][0]["t"] if nxt and nxt["kind"] == "episode" else fr[-1]["t"]   # contiguous, so durations add up
        out.append({
            "kind": "episode", "start_ms": fr[0]["t"], "end_ms": end, "duration_s": round((end - fr[0]["t"]) / 1000),
            "place": run["place"], "activity": run["activity"],
            "objects": sorted(n for n, c in counts.items() if c >= need),
            "held": sorted(n for n, c in held.items() if c >= need),
            "key_frame_ms": key["t"], "n_frames": len(fr),
        })
    return out


PEOPLE_MERGE_S = 20   # a person out of view for less than this still counts as the same sighting


def people_sightings(frames: list[dict]) -> list[dict]:
    """Presence only: stretches of time when at least one other person was in view. No identity, no faces matched.
    A sighting needs 2+ frames, so a single-frame misread never becomes 'you saw someone'."""
    runs, cur = [], None
    for f in frames:
        if f.get("people", 0) <= 0:
            continue
        if cur and f["t"] - cur[-1]["t"] <= PEOPLE_MERGE_S * 1000:
            cur.append(f)
        else:
            cur = [f]
            runs.append(cur)
    out = []
    for fr in runs:
        if len(fr) < 2:
            continue
        peak = max(f["people"] for f in fr)
        mid = (fr[0]["t"] + fr[-1]["t"]) / 2
        cands = [f for f in fr if f["people"] == peak and f["path"]] or fr
        key = min(cands, key=lambda f: abs(f["t"] - mid))
        places = [f["place"] for f in fr if f["place"]]
        out.append({"start_ms": fr[0]["t"], "end_ms": fr[-1]["t"], "max_people": peak,
                    "place": max(set(places), key=places.count) if places else None,
                    "key_frame_ms": key["t"], "n_frames": len(fr)})
    return out


def place_visits(frames: list[dict]) -> list[dict]:
    """Where the wearer was, WITHOUT the 30 s flicker-absorbing that episodes do, so a short trip outside survives.
    A visit needs 2+ frames (one-frame misreads dropped); neighbouring visits to the same place and setting merge."""
    runs, last_place = [], None
    for f in frames:
        place = f["place"] or last_place
        last_place = place or last_place
        key = (place, f["setting"])
        if runs and runs[-1]["key"] == key:
            runs[-1]["frames"].append(f)
        else:
            runs.append({"key": key, "frames": [f]})
    keep = [r for r in runs if len(r["frames"]) >= 2]
    merged = []
    for r in keep:
        if merged and merged[-1]["key"] == r["key"]:
            merged[-1]["frames"] += r["frames"]
        else:
            merged.append(r)
    out = []
    for r in merged:
        fr = r["frames"]
        mid = (fr[0]["t"] + fr[-1]["t"]) / 2
        key_f = min([f for f in fr if f["path"]] or fr, key=lambda f: abs(f["t"] - mid))
        signs = sorted({t for f in fr for t in f["texts"]})
        out.append({"start_ms": fr[0]["t"], "end_ms": fr[-1]["t"], "place": r["key"][0], "setting": r["key"][1],
                    "signs": signs[:5], "key_frame_ms": key_f["t"], "n_frames": len(fr)})
    return out


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()


def text_sightings(frames: list[dict]) -> list[dict]:
    """Legible text the camera read (shop names, signs). The model spells the same sign several ways and often reads only
    part of it ("BLACK SHEEP COFFEE", "Black Sheep", "SHEEP"), so fragments are folded into the longest string that contains
    them before counting. A sign needs 2+ frames to count, so a one-frame misread never becomes a fact."""
    by_key = {}
    for f in frames:
        for t in set(f["texts"]):
            k = _norm_text(t)
            if len(k) >= 3:
                by_key.setdefault(k, {"forms": {}, "frames": {}})
                by_key[k]["forms"][t] = by_key[k]["forms"].get(t, 0) + 1
                by_key[k]["frames"][f["t"]] = f
    keys = sorted(by_key, key=len, reverse=True)
    groups = {}                                   # longest key -> merged entry
    for k in keys:
        host = next((g for g in groups if len(k) >= 4 and k in g), None)
        if host is None:
            groups[k] = {"forms": dict(by_key[k]["forms"]), "frames": dict(by_key[k]["frames"])}
        else:
            for form, n in by_key[k]["forms"].items():
                groups[host]["forms"][form] = groups[host]["forms"].get(form, 0) + n
            groups[host]["frames"].update(by_key[k]["frames"])
    out = []
    for g in groups.values():
        fr = sorted(g["frames"].values(), key=lambda f: f["t"])
        if len(fr) < 2:
            continue
        longest = max(g["forms"], key=lambda t: (len(_norm_text(t)), g["forms"][t]))
        mid = (fr[0]["t"] + fr[-1]["t"]) / 2
        key_f = min([f for f in fr if f["path"]] or fr, key=lambda f: abs(f["t"] - mid))
        places = [f["place"] for f in fr if f["place"]]
        settings = [f["setting"] for f in fr if f["setting"]]
        out.append({"text": longest, "first_ms": fr[0]["t"], "last_ms": fr[-1]["t"], "n_frames": len(fr),
                    "place": max(set(places), key=places.count) if places else None,
                    "setting": max(set(settings), key=settings.count) if settings else None,
                    "key_frame_ms": key_f["t"]})
    return sorted(out, key=lambda x: (-x["n_frames"], x["first_ms"]))


def build_visits() -> list[dict]:
    return place_visits(_load_frames())


def build_texts() -> list[dict]:
    return text_sightings(_load_frames())


def build_people() -> list[dict]:
    return people_sightings(_load_frames())


def build() -> list[dict]:
    return segment(_load_frames())


def overview(eps: list[dict]) -> dict:
    """Totals the agent can quote: time per activity and place, and how much of the span the camera actually covered."""
    only = [e for e in eps if e["kind"] == "episode"]
    gaps = [e for e in eps if e["kind"] == "gap"]
    by_act, by_place = {}, {}
    for e in only:
        by_act[e["activity"]] = by_act.get(e["activity"], 0) + e["duration_s"]
        by_place[e["place"] or "unknown"] = by_place.get(e["place"] or "unknown", 0) + e["duration_s"]
    return {
        "first_ms": eps[0]["start_ms"] if eps else None,
        "last_ms": eps[-1]["end_ms"] if eps else None,
        "seconds_by_activity": by_act,
        "seconds_by_place": by_place,
        "seconds_with_no_footage": round(sum(g["end_ms"] - g["start_ms"] for g in gaps) / 1000),
        "gap_count": len(gaps),
    }
