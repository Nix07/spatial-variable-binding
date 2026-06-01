"""Aggregate per-cell baseline results into the Table 1 comparison.

Reads `results/<model>_<task>_baseline.{pkl,txt}` files, extracts the overall
accuracy from each, and emits `results/table1_summary.csv` plus a human-readable
`results/table1_summary.md`.

The markdown has two tables: the paper reproduction for the three Table-1 models
(`PAPER_MODELS`, compared to the paper's reported numbers with an OK/HIGH/LOW
flag), and an "added models" table for VLMs we ran beyond the paper
(`EXTRA_MODELS` — LLaVA-1.5-7B, Qwen2-VL-2B), which have no paper entry and so
show accuracy only.
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PAPER_TABLE1 = {
    ("qwen2vl", "squares"):  1.00,
    ("qwen2vl", "shapes"):   1.00,
    ("qwen2vl", "objects"):  0.73,
    ("qwen2vl", "whatsup"):  0.73,
    ("gemma3",  "squares"):  0.98,
    ("gemma3",  "shapes"):   0.99,
    ("gemma3",  "objects"):  0.99,
    ("gemma3",  "whatsup"):  0.62,
    ("pixtral", "squares"):  0.60,
    ("pixtral", "shapes"):   0.55,
    ("pixtral", "objects"):  0.63,
    ("pixtral", "whatsup"):  0.43,
}

# Paper-reproduction models (have Table-1 targets) and added models (no paper
# entry — see README/this module's __doc__). The added models are reported in
# their own table with accuracy only, no ✓/⚠️ comparison.
PAPER_MODELS = ["qwen2vl", "gemma3", "pixtral"]
EXTRA_MODELS = []
TASKS = ["squares", "shapes", "objects", "whatsup"]

DISPLAY_NAME = {
    "qwen2vl":   "Qwen2-VL-7B-Instruct",
    "gemma3":    "Gemma-3-4b-it",
    "pixtral":   "Pixtral-12b",
}


def _extract_from_pkl(path: Path) -> tuple[float, int | None, int | None]:
    """Return (accuracy, n_correct, n_total). Counts may be None for whatsup."""
    d = pickle.loads(path.read_bytes())
    if "overall_acc" in d:
        return float(d["overall_acc"]), d.get("total_correct"), d.get("total_examples")
    if "weighted_accuracy" in d:
        return float(d["weighted_accuracy"]), d.get("total_correct"), d.get("total_examples")
    if {"total_correct", "total_examples"} <= d.keys():
        tc, tt = d["total_correct"], d["total_examples"]
        return tc / tt, tc, tt
    raise ValueError(f"Cannot extract accuracy from {path.name}: keys={list(d.keys())[:8]}")


_TXT_TOTAL = re.compile(r"^\s*total:\s*([0-9.]+)", re.MULTILINE | re.IGNORECASE)


def _extract_from_txt(path: Path, prefer_floor: bool = True) -> float:
    """Parse legacy saved-log summary. For whatsup we prefer the 'floor prompt' block.

    The format is several `<name>:\n...\ntotal: X.YZZ` blocks; we pick the
    'floor prompt' block when present, else the first total in the file.
    """
    text = path.read_text()
    if prefer_floor and "floor prompt" in text.lower():
        idx = text.lower().index("floor prompt")
        m = _TXT_TOTAL.search(text, idx)
        if m:
            return float(m.group(1))
    m = _TXT_TOTAL.search(text)
    if m:
        return float(m.group(1))
    # Some .txt artifacts already store the headline accuracy verbatim.
    for line in text.splitlines():
        if "OVERALL" in line or "overall" in line:
            nums = re.findall(r"([0-9]*\.?[0-9]+)", line)
            if nums:
                return float(nums[-1])
    raise ValueError(f"No total/overall accuracy in {path.name}")


def _row_for(model: str, task: str, paper: float | None) -> dict:
    """Build one summary row. ``paper=None`` => added model (accuracy only)."""
    pkl = RESULTS / f"{model}_{task}_baseline.pkl"
    txt = RESULTS / f"{model}_{task}_baseline.txt"
    if pkl.exists():
        acc, n_correct, n_total = _extract_from_pkl(pkl)
        source = "pkl"
    elif txt.exists():
        acc = _extract_from_txt(txt, prefer_floor=(task == "whatsup"))
        n_correct = n_total = None
        source = "txt"
    else:
        return {"model": model, "task": task, "paper": paper, "ours": None,
                "delta": None, "match": "MISSING", "n_correct": None,
                "n_total": None, "source": "missing"}
    if paper is None:
        delta, match = None, "N/A"
    else:
        delta = round(acc - paper, 4)
        match = "OK" if abs(delta) <= 0.04 else ("HIGH" if delta > 0 else "LOW")
    return {"model": model, "task": task, "paper": paper, "ours": round(acc, 4),
            "delta": delta, "match": match, "n_correct": n_correct,
            "n_total": n_total, "source": source}


def collect_rows():
    """Rows for the paper models (with comparison) followed by the added models."""
    rows = [_row_for(m, t, PAPER_TABLE1[(m, t)]) for m in PAPER_MODELS for t in TASKS]
    rows += [_row_for(m, t, None) for m in EXTRA_MODELS for t in TASKS]
    return rows


def write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "task", "ours", "paper",
                                          "delta", "match", "n_correct",
                                          "n_total", "source"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows, path: Path):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["task"]] = r
    header = ("| Model | " + " | ".join(t.capitalize() for t in TASKS) + " |\n"
              + "|---|" + "|".join(["---"] * len(TASKS)) + "|")

    lines = ["# Table 1 — behavioral analysis\n",
             "## Paper reproduction\n",
             "Each cell shows `ours / paper` and a status flag.",
             "`OK` = within 0.04 of paper, `HIGH`/`LOW` = beyond that band.\n",
             header]
    for model in PAPER_MODELS:
        cells = [f"**{DISPLAY_NAME[model]}**"]
        for t in TASKS:
            r = by_model.get(model, {}).get(t)
            if r is None or r["ours"] is None:
                cells.append("—")
            elif r["match"] == "OK":
                cells.append(f"{r['ours']:.3f} / {r['paper']:.2f} ✓")
            else:
                cells.append(f"{r['ours']:.3f} / {r['paper']:.2f} ⚠️ {r['delta']:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    rows = collect_rows()
    write_csv(rows, RESULTS / "table1_summary.csv")
    write_markdown(rows, RESULTS / "table1_summary.md")
    paper_rows = [r for r in rows if r["paper"] is not None]
    n_ok = sum(1 for r in paper_rows if r["match"] == "OK")
    print(f"Wrote {RESULTS/'table1_summary.csv'} and {RESULTS/'table1_summary.md'}")
    print(f"Reproduction: {n_ok}/{len(paper_rows)} paper cells match within ±0.04 "
          f"(+{sum(1 for r in rows if r['paper'] is None and r['ours'] is not None)} added-model cells)")
