"""Generate Paper 1 conceptual figures via gpt-image-2 on ruoli.dev.

Two figures:
  intuition_architecture_stack       L1-L4 architecture: LLM translator,
                                     PyMC engine, BeliefStore memory,
                                     Fisher info obs-selection
  intuition_nl_vs_structured_channel NL prose bottleneck vs structured JSON,
                                     MAE -87% and -56% numbers on bars
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error


from _meis_keys import GPT_KEY as API_KEY
from _meis_keys import RUOLI_BASE_URL
API_URL = f"{RUOLI_BASE_URL}/images/generations"
OUT_DIR = Path(__file__).parent / "figs"
MODEL = "gpt-image-2"


FIGURES = [
    {
        "name": "intuition_architecture_stack",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background. Four stacked horizontal layers from TOP "
            "to BOTTOM, each a labeled rounded rectangle in a different "
            "muted pastel color. "
            "TOP layer (light yellow): 'L4 Orchestration — LLM as "
            "translator' with small icon of a speech bubble next to a "
            "JSON bracket icon. "
            "SECOND layer (light blue): 'L3 Structure — graph-edit count "
            "|Δ|, Markov category'. "
            "THIRD layer (light green): 'L2 Metric — KL divergence, "
            "Fisher information'. "
            "BOTTOM layer (light gray): 'L1 Representation — PyMC belief "
            "network' with a small DAG icon (3 circles connected by "
            "arrows). "
            "Vertical arrows on the LEFT side labeled 'translate' going "
            "down from L4 to L1, and arrows on the RIGHT side labeled "
            "'posterior / explanation' going up from L1 to L4. "
            "Caption text below: 'MEIS architecture: each layer stays "
            "in its own role.' "
            "Pure technical diagram, NeurIPS publication quality, no "
            "photos, no human figures."
        ),
    },
    {
        "name": "intuition_nl_vs_structured_channel",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background. TWO parallel pipelines shown side by "
            "side with a vertical divider. "
            "LEFT pipeline titled 'NL prose channel'. Shows: a small "
            "LLM cube labeled 'LLM' emitting a curly speech bubble "
            "with prose text 'lambda should be around 4, the pattern "
            "seems Poisson...'. An arrow to a PyMC engine box labeled "
            "'PyMC sampler (parses prose)'. An arrow to a bar labeled "
            "'MAE = 181.6' (tall red bar). "
            "RIGHT pipeline titled 'Structured JSON channel'. Shows: "
            "LLM cube labeled 'LLM' emitting a rigid rectangle containing "
            "JSON text looking like {\"lambda\": 4.0, \"family\": \"poisson\"}. "
            "An arrow to a PyMC engine box labeled 'PyMC sampler (typed "
            "contract)'. An arrow to a bar labeled 'MAE = 23.4' (very "
            "short green bar). "
            "Below the two pipelines, caption: 'Peregrines baseline: "
            "structured channel cuts MAE 87 percent (181.6 → 23.4). "
            "Prose is the bottleneck.' "
            "Pure technical diagram, NeurIPS publication quality."
        ),
    },
]


def generate_one(fig: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fig['name']}.png"
    payload = {
        "model": MODEL,
        "prompt": fig["prompt"],
        "size": fig.get("size", "1024x1024"),
        "n": 1,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {fig['name']}:\n{err_body}") from e
    data = json.loads(body)
    if "data" not in data or not data["data"]:
        raise RuntimeError(f"unexpected response for {fig['name']}:\n{body[:500]}")
    item = data["data"][0]
    if item.get("b64_json"):
        out_path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        with urllib.request.urlopen(item["url"], timeout=120) as img_resp:
            out_path.write_bytes(img_resp.read())
    else:
        raise RuntimeError(f"no b64_json or url in response: {item!r}")
    elapsed = time.time() - t0
    print(f"  [{elapsed:5.1f}s] {out_path.name}  "
          f"({out_path.stat().st_size/1024:.0f} KB)")
    return out_path


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else None
    for fig in FIGURES:
        if which and fig["name"] not in which:
            continue
        print(f"Generating {fig['name']} ({fig.get('size', '1024x1024')})...")
        try:
            generate_one(fig, OUT_DIR)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(5)


if __name__ == "__main__":
    main()
