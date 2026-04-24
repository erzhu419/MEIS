"""Generate Paper 2 conceptual intuition figures via gpt-image-2 on ruoli.dev.

Two figures:
  intuition_web_of_belief   — orphan vs in-vocabulary hypothesis on a belief web
  intuition_D_decomposition — composite score D = D_KL + λ|Δ| additive decomposition

Output goes to paper/figs/.
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


API_KEY = "sk-w4kL9fnlcUWqMvo97OcTjKTiU6waq2EDWIXWl8KdE3fILFyf"
API_URL = "https://ruoli.dev/v1/images/generations"
OUT_DIR = Path(__file__).parent / "figs"
MODEL = "gpt-image-2"


FIGURES = [
    {
        "name": "intuition_web_of_belief",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background, black and dark-blue ink. A belief network "
            "shown as nodes and directed edges arranged in a connected "
            "web shape in the center, labeled 'existing belief network B'. "
            "Five nodes in the center cluster labeled 'height', 'density', "
            "'weight', 'foot area', 'pressure', connected by curved "
            "arrows. "
            "On the LEFT side, a small green rounded rectangle labeled "
            "'hypothesis h_in-vocab (e.g. denser)' with a blue arrow "
            "pointing into the 'density' node — shows the hypothesis "
            "slots cleanly into the existing web. "
            "On the RIGHT side, two floating nodes labeled 'zodiac_A' "
            "and 'zodiac_C' shown in red with a red dashed boundary box "
            "and label 'orphan hypothesis h_orphan: adds 2 disconnected "
            "nodes, no causal path to the target'. Arrows do NOT connect "
            "the orphan nodes to any existing node. "
            "Below the whole figure, caption text: 'pure KL alone scores "
            "orphan first (near-zero drift); D = D_KL + λ|Δ| ranks orphan "
            "last.' "
            "Academic paper figure style like Distill.pub or NeurIPS, no "
            "photorealistic elements, pure technical diagram."
        ),
    },
    {
        "name": "intuition_D_decomposition",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, horizontal layout, minimal "
            "vector-graphics style, white background. Black and dark-blue "
            "ink. "
            "Four candidate claims stacked vertically on the LEFT as "
            "small labeled rounded rectangles (in dark gray ink): "
            "'h_3: larger feet', 'h_2: denser', 'h_1: taller', "
            "'h_orphan: zodiac'. "
            "In the CENTER, TWO vertical columns of small colored bars, "
            "one for each claim. The LEFT column (blue bars, labeled at "
            "top 'KL drift D_KL') is nearly empty for all claims. "
            "The RIGHT column (red bars, labeled at top 'structural "
            "penalty λ|Δ|') is zero for h_3, h_2, h_1 and very tall for "
            "h_orphan. "
            "In the RIGHT portion, stacked sum bars labeled 'D = D_KL + "
            "λ|Δ|', ranked from bottom (smallest, most coherent) to top "
            "(largest, least coherent): h_3 smallest, h_2, h_1, h_orphan "
            "largest. An arrow on the right pointing downward labeled "
            "'smaller D = more coherent'. "
            "Title at the top: 'D(h, B) = KL drift + structural penalty'. "
            "NeurIPS publication quality, pure diagram, no photos."
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
        time.sleep(5)   # small pacing buffer between images


if __name__ == "__main__":
    main()
