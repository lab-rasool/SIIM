#!/usr/bin/env python3
"""
extract_pathology.py  —  SIIM Local LLMs Learning Lab, Lab 3 (Pathology)

Goal: turn a free-text pathology report into structured, query-ready JSON
(diagnosis, site, stage, ...) using a LOCAL model. We then validate that the
model returned well-formed JSON with the fields we expect.

Like the radiology script, this only talks to your local Ollama server.
The synthetic report text never leaves the machine.

Examples
--------
    # Extract structured data from every synthetic pathology report
    python extract_pathology.py

    # One report, and save the structured output to a file
    python extract_pathology.py --report data/pathology/path_lung_002.txt --out lung.json

    # Use a larger model for tougher extraction
    python extract_pathology.py --model qwen2.5:7b

All data under data/pathology/ is SYNTHETIC. No PHI.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.ollama_client import DEFAULT_HOST, chat, ensure_model  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pathology")

# Fields we ask the model to populate. Use null when not stated in the report.
SCHEMA_FIELDS = [
    "primary_diagnosis",
    "primary_site",
    "laterality",
    "histologic_type",
    "grade",
    "tumor_size_cm",
    "margins",
    "lymphovascular_invasion",
    "stage",
    "biomarkers",
]

SYSTEM_PROMPT = (
    "You are a precise pathology information-extraction engine. You read a "
    "free-text pathology report and return a single JSON object. You only use "
    "information explicitly stated in the report. If a field is not stated, you "
    "set it to null. You never guess or fabricate values."
)

USER_TEMPLATE = """Extract the following fields from the pathology report and return ONE JSON object.

Required keys (use null if a value is not stated in the report):
- primary_diagnosis: string  (e.g. "invasive ductal carcinoma")
- primary_site: string       (organ/site, e.g. "breast", "lung", "colon")
- laterality: string or null ("left", "right", or null)
- histologic_type: string or null
- grade: string or null      (e.g. "Nottingham grade 2", "Gleason 3+4=7")
- tumor_size_cm: number or null
- margins: string or null    ("negative", "positive", or a short description)
- lymphovascular_invasion: "present", "absent", or null
- stage: string or null      (e.g. "pT3 pN1b")
- biomarkers: object or null  (e.g. {{"ER": "positive", "HER2": "negative"}})

Return ONLY the JSON object. No prose, no markdown.

PATHOLOGY REPORT:
\"\"\"
{report}
\"\"\"
"""


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


def validate(obj: dict) -> list[str]:
    """Return a list of validation warnings (empty list == clean)."""
    warnings = []
    if not isinstance(obj, dict):
        return ["Top-level value is not a JSON object."]
    for field in SCHEMA_FIELDS:
        if field not in obj:
            warnings.append(f"missing key: {field}")
    if not obj.get("primary_diagnosis"):
        warnings.append("primary_diagnosis is empty/null (expected a value)")
    size = obj.get("tumor_size_cm")
    if size is not None and not isinstance(size, (int, float)):
        warnings.append(f"tumor_size_cm should be a number or null, got {type(size).__name__}")
    return warnings


def extract(report: str, model: str, host: str) -> tuple[dict | None, str]:
    prompt = USER_TEMPLATE.format(report=report)
    raw = chat(prompt, model=model, system=SYSTEM_PROMPT, host=host, json_mode=True, temperature=0.0)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured JSON from synthetic pathology reports locally.")
    ap.add_argument("--report", help="Path to a single report .txt (default: all in data/pathology/)")
    ap.add_argument("--model", default="llama3.2", help="Ollama model tag (default: llama3.2)")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    ap.add_argument("--out", help="Optional path to write a combined JSON array of results")
    args = ap.parse_args()

    ensure_model(args.model, host=args.host)
    reports = load_reports(args.report)
    print(f"\nLoaded {len(reports)} synthetic pathology report(s). Model: {args.model}. Host: {args.host}\n")

    results = []
    for name, text in reports:
        print("=" * 78)
        print(f"REPORT: {name}")
        print("=" * 78)
        obj, raw = extract(text, args.model, args.host)
        if obj is None:
            print("[!] Model did not return valid JSON. Raw output:\n")
            print(raw)
            results.append({"_source": name, "_error": "invalid_json", "_raw": raw})
            print()
            continue

        print(json.dumps(obj, indent=2))
        warnings = validate(obj)
        if warnings:
            print("\n  VALIDATION WARNINGS:")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("\n  ✓ Validation passed: all expected fields present and well-typed.")
        obj["_source"] = name
        results.append(obj)
        print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} record(s) to {args.out}")


if __name__ == "__main__":
    main()
