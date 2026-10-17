"""Perceiver: one keyframe in, structured scene description out (NVIDIA Cosmos via OpenAI-compatible API)."""
import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Nemotron VL on CoreWeave inference (~3s/frame) vs NVIDIA free tier (~30s/frame, same model).
# To use the NVIDIA tier instead: PERCEIVER_BASE_URL=https://integrate.api.nvidia.com/v1,
# PERCEIVER_KEY_ENV=NVIDIA_KEY, PERCEIVER_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
MODEL = os.getenv("PERCEIVER_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B")
BASE_URL = os.getenv("PERCEIVER_BASE_URL", "https://api.inference.wandb.ai/v1")
KEY_ENV = os.getenv("PERCEIVER_KEY_ENV", "COREWEAVE_KEY")

PROMPT = """You are the eyes of a wearable memory device. The image is a first-person view from the wearer.
Describe only what is clearly visible. Return ONLY a JSON object, no prose, with this shape:
{
  "location": "room or place, e.g. kitchen counter, sofa, desk",
  "action": "what the wearer is doing in one short phrase",
  "people_visible": 0,
  "setting": "indoor | outdoor",
  "visible_text": [],
  "activity": "exactly one of: working | walking | eating_drinking | using_phone | talking | resting | other",
  "objects": [{"name": "singular lowercase noun, e.g. keys, phone, mug", "state": "held | on_surface | in_container | worn | unknown"}],
  "relations": [{"subj": "object name", "pred": "holding | on | in | near | picked_up | put_down", "obj": "object or surface name"}]
}
"visible_text" lists names or words you can clearly READ on real signs, shopfronts, menus, labels or street signs in the scene
(exact spelling, e.g. "Costa Coffee"); [] if none. Ignore text on computer or phone screens and do not guess unreadable text.
"people_visible" is how many OTHER real people are physically in view right now (integer, 0 if none). Do not count the
wearer's own hands, reflections, photos, posters or people shown on a screen.
Use "wearer" as subj when the wearer's hands hold something. Only list everyday objects a person might lose or use
(keys, phone, wallet, bag, mug, laptop, glasses, etc.), not walls or floors."""

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=os.environ[KEY_ENV], timeout=60)
    return _client


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def describe(jpeg_path: str | Path, attempts: int = 2) -> dict:
    for i in range(attempts):
        try:
            return _describe_once(jpeg_path)
        except (ValueError, json.JSONDecodeError):  # truncated / malformed output: sampling is random, try again
            if i == attempts - 1:
                raise


def _describe_once(jpeg_path: str | Path) -> dict:
    b64 = base64.b64encode(Path(jpeg_path).read_bytes()).decode()
    resp = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=5000,  # reasoning models spend tokens thinking before the JSON (2500 truncated ~10% of frames)
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError(f"empty model output (finish_reason={resp.choices[0].finish_reason})")
    return _extract_json(content)


if __name__ == "__main__":
    import sys
    print(json.dumps(describe(sys.argv[1]), indent=2))
