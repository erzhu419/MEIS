"""Generate Paper 3 conceptual figures via gpt-image-2 on ruoli.dev.

Two figures:
  intuition_signature_stack   Four independent signature layers all
                              converging on the same 3-class partition
                              (ARI = 1.00)
  intuition_transfer_protocol Algorithm 1 visualised: source rich data
                              → posterior σ → gate → target prior SD
"""

from __future__ import annotations

import base64
import json
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
        "name": "intuition_signature_stack",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, minimal vector-graphics style, "
            "white background. Four parallel horizontal rows, each "
            "representing one equivalence-detection layer; each row "
            "starts on the LEFT with a small labeled rectangle and ends "
            "on the RIGHT with three color-coded circular clusters "
            "corresponding to the three ground-truth equivalence classes. "
            "Row 1 (orange): 'Op-multiset (PyTensor bag of ops)' → "
            "three clusters showing 4 blue dots, 3 green dots, 3 purple "
            "dots. "
            "Row 2 (teal): 'Weisfeiler-Lehman (DAG subtree refinement)' "
            "→ same 4/3/3 cluster arrangement. "
            "Row 3 (light blue): 'Markov category (symbolic string "
            "diagram)' → same 4/3/3. "
            "Row 4 (salmon): 'BSS + Perrone (semantic likelihood "
            "check)' → same 4/3/3. "
            "To the FAR RIGHT, a single large annotation 'ARI = 1.00 "
            "(all four layers agree)' with a green check mark. "
            "The rows are labeled on the LEFT and the clusters on the "
            "RIGHT show domain names in small text "
            "(rc_circuit/radioactive_decay/first_order/forgetting in the "
            "exp_decay cluster; capacitor/monomolecular/light_adapt in "
            "saturation; rlc/pendulum/mass_spring in damped). "
            "Title at TOP: 'Four independent signature layers recover "
            "the same 3-class partition.' "
            "NeurIPS publication quality, pure diagram, no people, "
            "no photos."
        ),
    },
    {
        "name": "intuition_transfer_protocol",
        "size": "1024x1024",
        "prompt": (
            "Clean academic paper figure, vertical layout, minimal "
            "vector-graphics style, white background. "
            "Shows a 4-step transfer protocol as a vertical flow. "
            "Step 1 (top, light blue box): 'Source domain S — rich data "
            "(30 observations)' with a small tight Gaussian curve labeled "
            "'σ_log^S = 0.014 (narrow posterior)'. "
            "Arrow pointing RIGHT with label 'extract σ'. "
            "Step 2 (middle, gray diamond gate shape): "
            "'Gate: sig(T) = sig(S)?' with YES arrow going down and "
            "NO arrow on the right with red X mark and label "
            "'raise ValueError'. "
            "Step 3 (light green box): 'Target domain T — 3 observations "
            "only'. Below it, two side-by-side small plots: "
            "LEFT labeled 'cold-start prior σ = 1.5' (a wide Gaussian), "
            "RIGHT labeled 'transferred prior σ = σ_log^S' (a narrow "
            "Gaussian). "
            "Step 4 (bottom, salmon box): 'Fit NUTS, report held-out "
            "MSE' with two bars: big red 'cold = 0.359', tiny green "
            "'transfer = 0.011'. Label: '+97% MSE reduction'. "
            "Title at TOP: 'Structural-transfer protocol with "
            "signature gate (Algorithm 1).' "
            "NeurIPS publication quality, pure technical diagram."
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
