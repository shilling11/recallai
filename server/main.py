"""RecallAI ingest server: receives gated frames from the phone and stores them as keyframes."""
import io
import json
import shutil
from contextlib import asynccontextmanager
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import edge_tts
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from PIL import Image
from pydantic import BaseModel

from . import agent, episodes, janitor, memory, pipeline

ROOT = Path(__file__).resolve().parent.parent
KEYFRAMES = ROOT / "data" / "keyframes"

KEYFRAMES.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app):
    """Every real server start is a brand-new session: wipe the previous session's keyframes and memory.
    Done here and not at import time, so importing this module (tests, tooling) can never delete data."""
    shutil.rmtree(KEYFRAMES, ignore_errors=True)
    for db_file in KEYFRAMES.parent.glob("recall.db*"):   # also catches -wal / -journal files
        db_file.unlink(missing_ok=True)
    KEYFRAMES.mkdir(parents=True, exist_ok=True)
    janitor.start()
    yield


NO_CACHE = {"Cache-Control": "no-store"}   # Safari otherwise keeps showing an old copy of the page
SESSION_ID = int(time.time() * 1000)
_events_built_at = 0.0
app = FastAPI(lifespan=lifespan)
stats = {"received": 0, "sent_by_phone": 0, "seen_by_phone": 0}


@app.get("/")
def phone_page():
    return FileResponse(ROOT / "web" / "phone.html", headers=NO_CACHE)


@app.get("/stats")
def get_stats():
    return {**stats, "pipeline": pipeline.stats, "memory": pipeline.freshness(), "storage": janitor.storage_stats()}


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):
    try:
        result = agent.ask(body.question)
    except Exception as e:   # the phone must always get an answer it can show, never a bare 500
        print("ask failed:", repr(e))
        return {"answer": "Sorry, I couldn't get an answer just now. Please try again.", "evidence": [], "proof": []}
    # Proof: one slide per cited event, oldest first, for the carousel
    result["proof"] = [
        {"t_ms": e["t_ms"], "url": f"/keyframes/{e['t_ms']}.jpg", "caption": e["caption"]} for e in result["evidence"]
    ][:12]
    return result


@app.get("/tts")
async def tts(text: str):
    """Speak an answer. Generated on the laptop because iOS Safari often blocks speechSynthesis."""
    audio = b""
    async for chunk in edge_tts.Communicate(text[:400], "en-GB-SoniaNeural").stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return Response(audio, media_type="audio/mpeg")


@app.get("/episodes")
def get_episodes():
    eps = episodes.build()
    return {"overview": episodes.overview(eps), "episodes": eps}


@app.get("/graph")
def graph(t: int | None = None):
    global _events_built_at
    if time.time() - _events_built_at > 3:   # replay calls this ~6x a second; markers only need to be a few seconds fresh
        _events_built_at = time.time()
        memory.rebuild_events()
    return memory.graph_at(t)


@app.get("/graph/series")
def graph_series(n: int = 120):
    memory.rebuild_events()
    return memory.graph_series(n)


_thumb_cache: dict[int, bytes] = {}


@app.get("/thumb/{t_ms}.jpg")
def thumb(t_ms: int):
    """Small preview of a keyframe for the scene replay (full frames are too heavy to swap six times a second over a phone link)."""
    data = _thumb_cache.get(t_ms)
    if data is None:
        path = KEYFRAMES / f"{t_ms}.jpg"
        if not path.exists():
            return Response(status_code=404)
        im = Image.open(path)
        im.thumbnail((560, 560))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=72)
        data = buf.getvalue()
        if len(_thumb_cache) > 500:
            _thumb_cache.pop(next(iter(_thumb_cache)))
        _thumb_cache[t_ms] = data
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/session")
def session():
    """Changes on every server start, so the phone can drop a chat history whose frames no longer exist."""
    return {"id": SESSION_ID}


@app.get("/app")
def talk_page():
    return FileResponse(ROOT / "web" / "app.html", headers=NO_CACHE)


@app.websocket("/ws/frames")
async def frames(ws: WebSocket):
    """Protocol: text message {"t": ms, "seen": n, "sent": n} followed by a binary JPEG."""
    await ws.accept()
    meta = None
    try:
        while True:
            msg = await ws.receive()
            if msg.get("text") is not None:
                meta = json.loads(msg["text"])
            elif msg.get("bytes") is not None:
                t_ms = int((meta or {}).get("t", time.time() * 1000))
                stats["received"] += 1
                pipeline.submit(t_ms, msg["bytes"], KEYFRAMES / f"{t_ms}.jpg")
                await ws.send_text("k")   # ack: lets the phone show whether frames are really reaching the laptop
                if meta:
                    stats["seen_by_phone"] = meta.get("seen", 0)
                    stats["sent_by_phone"] = meta.get("sent", 0)
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass


app.mount("/keyframes", StaticFiles(directory=KEYFRAMES), name="keyframes")
