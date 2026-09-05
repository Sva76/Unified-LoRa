"""Reanalyze Test 7 without training; preserve raw logs and both timing conventions.

Usage: python validation/test7_reanalyze.py [log.json] [--output new-report.json]
Without --output this command only prints a report. The historical predictive
verdict remains void because of the task confounds in the correction note.
"""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .test7_metrics import summarize
    from .phi_utils import write_run_json
except ImportError:
    from test7_metrics import summarize
    from phi_utils import write_run_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", type=Path,
                    default=Path(__file__).with_name("phi_lead_time_log.json"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.output and (args.output.resolve() == args.log.resolve() or args.output.exists()):
        ap.error("output must be a new file, distinct from the source log")
    raw = args.log.read_bytes()
    report = summarize(json.loads(raw))
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.output:
        write_run_json(args.output, report)


if __name__ == "__main__":
    main()
