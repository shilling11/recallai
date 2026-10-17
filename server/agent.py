"""Agent: answers questions about the wearer's day by calling memory tools, and cites evidence frames."""
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from . import episodes, memory

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Reasoning + tool calling on CoreWeave inference. Any OpenAI-compatible endpoint works via the env vars.
MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
BASE_URL = os.getenv("AGENT_BASE_URL", "https://api.inference.wandb.ai/v1")
KEY_ENV = os.getenv("AGENT_KEY_ENV", "COREWEAVE_KEY")
_client = None
EFFORT = os.getenv("AGENT_EFFORT", "low")   # "low" roughly halves answer time; empty replies are retried at normal effort

SYSTEM = """You are the memory of a wearable camera. The wearer asks about things they did or saw earlier today.
Answer ONLY from tool results. Never guess. If the tools return nothing for an object, say you have no record of it.
Keep answers to one or two short sentences (more only for day summaries), spoken style, and give times as local clock time (e.g. 2:03 PM).
Prefer the confirmed event; mention a newer unconfirmed sighting only if it differs, and hedge it ("I just saw ...").
For "what did I do" style questions, call the episodes tool: it lists what the wearer was doing, where and for how long,
plus totals and how many seconds had no footage. Narrate the episodes in time order in at most 4 short spoken sentences,
say "from the footage I have", and mention a gap in the footage if there is one. Never invent activities between episodes.
Say an object was "held" only if it is in that episode's held list; otherwise say it was "there" or "on the desk".
Report what was seen, not what it implies: "holding a mug", not "so you drank something".
When citing for a summary, cite one episode id per main stretch of the day (at most 6).
You keep track of objects (picked up, put down, worn), the wearer's overall activity and place, and only WHETHER and
WHEN other people were in view and how many. You do NOT know who anyone is, what they said, how they felt, or whether the
wearer spoke to them: for those questions say so, and offer only what you do know (that someone was in view, and when).
For "was I outside" / "where did I go" questions call the places tool; for names on shops or signs call text_seen. If text_seen
finds nothing, say you did not read a name, and never guess one from the kind of place.
For "did I see anyone" / "who was around" questions call the people tool. Never describe or name individuals, and never say what they were doing ("passed by", "stood") - only that they were in view. You do NOT track people, faces, conversations, emotions or
anything else; if asked about those, say you only track objects so far, and do not guess.
Always call a tool before answering. Never mention ids, field names or timestamps in milliseconds in your answer.
Proof: the screen shows a picture for each event you cite. Finish your reply with one last line "CITE: e1,e3" listing the
ids of ONLY the events your answer actually relies on, or "CITE: none" if the answer needs no picture (for example when
there is no record). Never cite an event that is not directly part of your answer."""

_SPECS = [
    ("last_seen", "Latest recorded event for one object (e.g. keys, phone, wallet): what happened and where, with time.",
     {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}),
    ("events", "Chronological list of recorded events (picked up, put down, worn). Optionally for one object and/or from N minutes ago.",
     {"type": "object", "properties": {"object": {"type": "string"}, "minutes_ago": {"type": "number"}}}),
    ("places", "Where the wearer was over time, indoor or outdoor, including short visits, with any shop or sign names read there. "
     "Use for 'was I outside', 'where did I go', 'which places did I visit'. Optional filter: minutes_ago.",
     {"type": "object", "properties": {"minutes_ago": {"type": "number"}}}),
    ("text_seen", "Legible text the camera read on signs, shopfronts, menus or labels (e.g. a shop name), with when and where. "
     "Use for 'what was the name of the cafe', 'what did the sign say'. Optional: contains (substring filter).",
     {"type": "object", "properties": {"contains": {"type": "string"}}}),
    ("people", "Times when other people were in view (presence and head-count only, no identities), with place. "
     "Use for 'did I see anyone', 'was anyone around', 'when were people nearby'. Optional filter: minutes_ago.",
     {"type": "object", "properties": {"minutes_ago": {"type": "number"}}}),
    ("episodes", "What the wearer was doing, where and for how long, as time-ordered episodes, with totals per activity/place "
     "and seconds with no footage. Use for questions like 'what did I do today', 'how long was I at my desk', 'where was I earlier'. "
     "Optional filters: minutes_ago, place, activity (working, walking, eating_drinking, using_phone, talking, resting, other).",
     {"type": "object", "properties": {"minutes_ago": {"type": "number"}, "place": {"type": "string"}, "activity": {"type": "string"}}}),
]
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": p}} for n, d, p in _SPECS]


def _clock(t_ms: int) -> str:
    return time.strftime("%I:%M:%S %p", time.localtime(t_ms / 1000)).lstrip("0")


_ACT_LABEL = {"eating_drinking": "eating or drinking", "using_phone": "using the phone"}


def _clock_short(t_ms: int) -> str:
    return time.strftime("%I:%M %p", time.localtime(t_ms / 1000)).lstrip("0")


def _run_tool(name: str, args: dict, registry: dict):
    """Run a tool. Every event it returns gets a short id (e1, e2, ...) that the model can cite;
    `registry` maps each id to the frame time that proves it."""
    def tag(t_ms: int, caption: str) -> str:
        eid = f"e{len(registry) + 1}"
        registry[eid] = {"t_ms": t_ms, "caption": caption}
        return eid

    if name == "last_seen":
        e = memory.last_seen(args["object"])
        if not e:
            return {"result": "no record of this object"}
        out = {}
        if "summary" in e:
            out = {"id": tag(e["frame_t_ms"] or e["t_ms"], e["summary"]), "summary": e["summary"], "time": _clock(e["t_ms"])}
        s = e.get("latest_sighting")
        if s:
            out["newer_unconfirmed_sighting"] = {
                "id": tag(s["t_ms"], f'{args["object"]}: just seen {s["place"]} (unconfirmed)'), "state": s["state"], "place": s["place"], "time": _clock(s["t_ms"]),
                "note": "seen in a single frame only; say 'I just saw' and hedge",
            }
        return out
    if name == "events":
        since = int(time.time() * 1000 - args["minutes_ago"] * 60000) if args.get("minutes_ago") else None
        evs = memory.events_for(args.get("object"), since)
        return [{"id": tag(e["t_ms"], e["summary"]), "summary": e["summary"], "time": _clock(e["t_ms"])} for e in evs] or {"result": "no events"}
    if name == "places":
        visits = episodes.build_visits()
        if args.get("minutes_ago"):
            cutoff = int(time.time() * 1000 - args["minutes_ago"] * 60000)
            visits = [v for v in visits if v["end_ms"] >= cutoff]
        if not visits:
            return {"result": "no places recorded yet"}
        out_v = []
        for v in visits[-25:]:
            span = _clock_short(v["start_ms"]) if v["end_ms"] - v["start_ms"] < 60000 else f'{_clock_short(v["start_ms"])} to {_clock_short(v["end_ms"])}'
            where = f'{v["place"] or "unknown place"} ({v["setting"] or "setting unknown"})'
            caption = f"{span} · {where}" + (f' · sign: {", ".join(v["signs"])}' if v["signs"] else "")
            out_v.append({"id": tag(v["key_frame_ms"], caption), "time": span, "place": v["place"], "setting": v["setting"], "signs_read": v["signs"]})
        return {"visits": out_v}
    if name == "text_seen":
        texts = episodes.build_texts()
        if not texts:
            return {"result": "no legible text such as a shop name or sign was read in the footage I have"}
        note = None
        if args.get("contains"):
            hits = [t for t in texts if args["contains"].lower() in t["text"].lower()]
            if hits:
                texts = hits
            else:   # a narrow filter must never hide text that was read: show everything instead
                note = f'nothing matched "{args["contains"]}"; showing all text that was read'
        out_t = [{"id": tag(t["key_frame_ms"], f'"{t["text"]}" · {_clock_short(t["first_ms"])}' + (f' ({t["place"]})' if t["place"] else "")),
                  "text": t["text"], "seen_in_frames": t["n_frames"], "setting": t["setting"], "place": t["place"],
                  "time": _clock_short(t["first_ms"])} for t in texts[:15]]
        res = {"text_read": out_t, "hint": "shop names usually come from outdoor or entrance scenes and appear in the most frames; "
                                           "text on laptops or packaging is not a shop name"}
        if note:
            res["note"] = note
        return res
    if name == "people":
        sightings = episodes.build_people()
        if args.get("minutes_ago"):
            cutoff = int(time.time() * 1000 - args["minutes_ago"] * 60000)
            sightings = [x for x in sightings if x["end_ms"] >= cutoff]
        if not sightings:
            return {"result": "no other people were in view in the footage I have"}
        out_s = []
        for x in sightings[-20:]:
            span = _clock_short(x["start_ms"]) if x["end_ms"] - x["start_ms"] < 60000 else f'{_clock_short(x["start_ms"])} to {_clock_short(x["end_ms"])}'
            who = "1 person" if x["max_people"] == 1 else f'{x["max_people"]} people'
            caption = f'{span} · {who} in view' + (f' ({x["place"]})' if x["place"] else "")
            out_s.append({"id": tag(x["key_frame_ms"], caption), "time": span, "people_in_view": x["max_people"], "place": x["place"]})
        return {"sightings": out_s, "note": "presence and head-count only; no identities, speech or emotions"}
    if name == "episodes":
        eps = episodes.build()
        if not eps:
            return {"result": "no footage described yet"}
        info = episodes.overview(eps)
        if args.get("minutes_ago"):
            cutoff = int(time.time() * 1000 - args["minutes_ago"] * 60000)
            eps = [e for e in eps if e["end_ms"] >= cutoff]
        if args.get("place"):
            eps = [e for e in eps if e["kind"] == "gap" or args["place"].lower() in (e["place"] or "")]
        if args.get("activity"):
            eps = [e for e in eps if e["kind"] == "gap" or e["activity"] == args["activity"].lower().replace(" ", "_")]
        out_eps = []
        for e in eps[-30:]:
            span = f'{_clock_short(e["start_ms"])} to {_clock_short(e["end_ms"])}'
            if e["kind"] == "gap":
                out_eps.append({"time": span, "no_footage_seconds": round((e["end_ms"] - e["start_ms"]) / 1000)})
                continue
            what = f'{_ACT_LABEL.get(e["activity"], e["activity"].replace("_", " "))} at {e["place"] or "an unknown place"}'
            caption = f'{span} · {what}' + (f' ({", ".join(e["objects"])})' if e["objects"] else "")
            out_eps.append({"id": tag(e["key_frame_ms"], caption), "time": span, "minutes": round(e["duration_s"] / 60, 1),
                            "place": e["place"], "activity": e["activity"], "objects": e["objects"], "held": e["held"]})
        return {
            "footage_from": _clock_short(info["first_ms"]), "footage_to": _clock_short(info["last_ms"]),
            "minutes_by_activity": {k: round(v / 60, 1) for k, v in info["seconds_by_activity"].items()},
            "minutes_by_place": {k: round(v / 60, 1) for k, v in info["seconds_by_place"].items()},
            "minutes_with_no_footage": round(info["seconds_with_no_footage"] / 60, 1),
            "episodes": out_eps,
        }
    return {"error": "unknown tool"}


_CITE = re.compile(r"CITE:\s*([^\n]*)", re.I)   # the model may put it inline or on its own line


def _split_cite(text: str, registry: dict):
    """Separate the spoken answer from the trailing CITE line; return (answer, [{t_ms, caption}] in time order)."""
    m = _CITE.search(text)
    answer = _CITE.sub("", text).strip()
    cited = []
    if m:
        for eid in re.findall(r"e\d+", m.group(1).lower()):
            if eid in registry and registry[eid]["t_ms"] not in {c["t_ms"] for c in cited}:
                cited.append(registry[eid])
    return answer, sorted(cited, key=lambda c: c["t_ms"])


def _create(messages, effort):
    kw = {"extra_body": {"reasoning_effort": effort}} if effort else {}
    return _client.chat.completions.create(model=MODEL, max_tokens=3000, tools=TOOLS, messages=messages, **kw)


def ask(question: str) -> dict:
    global _client
    _client = _client or OpenAI(base_url=BASE_URL, api_key=os.environ[KEY_ENV], timeout=60)
    memory.rebuild_events()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    registry = {}
    for _ in range(5):
        resp = _create(messages, effort=EFFORT)
        msg = resp.choices[0].message
        if not msg.tool_calls and not (msg.content or "").strip():
            # Low effort sometimes ends with no text at all. That reads as "no response" on the phone, so retry at normal effort.
            resp = _create(messages, effort=None)
            msg = resp.choices[0].message
        if not msg.tool_calls:
            answer, evidence = _split_cite(msg.content or "", registry)
            return {"answer": answer, "evidence": evidence}   # no CITE line means no pictures: never show unrelated frames
        messages.append(msg.model_dump(exclude_none=True, exclude={"reasoning_content", "reasoning"}))
        for tc in msg.tool_calls:
            out = _run_tool(tc.function.name, json.loads(tc.function.arguments or "{}"), registry)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    return {"answer": "Sorry, I could not work that out.", "evidence": []}


if __name__ == "__main__":
    import sys
    print(json.dumps(ask(" ".join(sys.argv[1:]) or "Where did I leave my keys?"), indent=2))
