#!/usr/bin/env python3
"""
summarize_report.py  —  SIIM Local LLMs Learning Lab, Lab 3 (Radiology)

Goal: turn a long free-text radiology report into a concise, structured
3-line impression using a model running ENTIRELY on local hardware.

Nothing in this script touches the internet. It talks only to your local
Ollama server, so the synthetic report text never leaves the machine —
which is the whole point of running LLMs behind the firewall.

Examples
--------
    # Summarize every synthetic report with the default small model
    python summarize_report.py

    # Summarize one specific report
    python summarize_report.py --report data/radiology/ct_chest_001.txt

    # Compare a small vs. a larger model side by side (slide: "Compare
    # small vs. larger model outputs side by side")
    python summarize_report.py --compare llama3.2 qwen2.5:7b

The original report text is printed above each impression so you can see the
input and the model's output together. Add --no-report to print only the output.

All data under data/radiology/ is SYNTHETIC. No PHI.
"""

import argparse
import glob
import os
import sys

# Allow running as `python summarize_report.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.ollama_client import DEFAULT_HOST, chat, ensure_model  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "radiology")

SYSTEM_PROMPT = (
    "You are a careful radiology assistant. You summarize free-text radiology "
    "reports for clinicians. You never invent findings that are not present in "
    "the report. You are concise and factual."
)

USER_TEMPLATE = """Summarize the following radiology report as a structured impression.

Rules:
- Output EXACTLY three numbered lines.
- Line 1: the single most important finding.
- Line 2: the second most important finding, or "No additional significant findings."
- Line 3: the recommended next step, if one is stated; otherwise "No specific follow-up recommended."
- Use only information present in the report. Do not add new findings.

REPORT:
\"\"\"
{report}
\"\"\"

STRUCTURED IMPRESSION:"""


def _section(title: str, body: str | None = None, width: int = 78) -> str:
    """Format a labeled block: a title, a rule beneath it, then optional body.

    Used to keep the lab output readable — the INPUT REPORT and each model's
    IMPRESSION print under their own clearly labeled heading.
    """
    parts = [title, "-" * width]
    if body is not None:
        parts.append(body.strip("\n"))
    return "\n".join(parts)


def load_reports(target: str | None) -> list[tuple[str, str]]:
    if target:
        if not os.path.exists(target):
            raise SystemExit(f"[!] Report not found: {target}")
        with open(target, encoding="utf-8") as f:
            return [(os.path.basename(target), f.read())]
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not paths:
        raise SystemExit(f"[!] No reports found in {DATA_DIR}")
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append((os.path.basename(p), f.read()))
    return out


def summarize(report: str, model: str, host: str) -> str:
    prompt = USER_TEMPLATE.format(report=report)
    return chat(prompt, model=model, system=SYSTEM_PROMPT, host=host, temperature=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize synthetic radiology reports locally.")
    ap.add_argument("--report", help="Path to a single report .txt (default: all in data/radiology/)")
    ap.add_argument("--model", default="llama3.2", help="Ollama model tag (default: llama3.2)")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("SMALL", "LARGE"),
        help="Two model tags to compare side by side, e.g. --compare llama3.2 qwen2.5:7b",
    )
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    ap.add_argument(
        "--no-report",
        action="store_true",
        help="Hide the input report text; print only the model's impression.",
    )
    args = ap.parse_args()

    models = args.compare if args.compare else [args.model]
    for m in models:
        ensure_model(m, host=args.host)

    reports = load_reports(args.report)
    show_report = not args.no_report
    print(f"\nLoaded {len(reports)} synthetic radiology report(s). Host: {args.host}\n")

    for name, text in reports:
        print("=" * 78)
        print(f"REPORT: {name}")
        print("=" * 78)

        if show_report:
            print()
            print(_section("INPUT REPORT", text))

        for m in models:
            print()
            try:
                body = summarize(text, m, args.host)
            except SystemExit as e:
                body = str(e)
            print(_section(f"IMPRESSION ({m})", body))
        print()


if __name__ == "__main__":
    main()
