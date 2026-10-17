"""Run the pipeline on the training videos and report jury-like metrics per camera layout."""
import argparse
import csv
import json
import time
from pathlib import Path

from app.config import load_settings
from app.pipeline.probes import load_facts
from app.pipeline.rules import load_rules
from app.pipeline.run import APP_DIR, analyze, build_models
from app.schemas import Violation
from scripts.labels import DUPLICATES, load_labels, video_key


def score(preds: dict[str, list[Violation]], labels: dict[str, list[float]],
          types: dict[tuple[str, float], str], layouts: dict[str, str]) -> dict:
    groups: dict[str, dict] = {}

    def bump(g, k, n=1):
        groups.setdefault(g, {"violation_videos": 0, "caught": 0, "clean_videos": 0, "false_alarms": 0,
                              "labels": 0, "time_within_2s": 0, "type_correct": 0, "abs_dt": []})
        if k == "abs_dt":
            groups[g][k].append(n)
        else:
            groups[g][k] += n

    for name, ts in labels.items():
        if name not in preds:
            continue
        p = preds[name]
        for g in ("all", layouts.get(name, "?")):
            if ts:
                bump(g, "violation_videos")
                bump(g, "caught", int(bool(p)))
                for t in ts:
                    bump(g, "labels")
                    if p:
                        near = min(p, key=lambda v: abs(v.timecode_sec - t))
                        bump(g, "abs_dt", abs(near.timecode_sec - t))
                        within = abs(near.timecode_sec - t) <= 2.0
                        bump(g, "time_within_2s", int(within))
                        bump(g, "type_correct", int(within and near.rule_id == types.get((name, t))))
            else:
                bump(g, "clean_videos")
                bump(g, "false_alarms", int(bool(p)))
    for g in groups.values():
        dts = sorted(g.pop("abs_dt"))
        g["median_abs_dt"] = round(dts[len(dts) // 2], 2) if dts else None
    return groups


def candidate_recall(details_by_video: dict[str, list[dict]], labels: dict[str, list[float]],
                     win: float = 3.0) -> float | None:
    """Share of labelled violations (on videos that were run) with any CV/S8 candidate within +-win s,
    accepted or not: the ceiling S7 and the decision logic can reach."""
    hit = n = 0
    for name, details in details_by_video.items():
        for t in labels.get(name, []):
            n += 1
            hit += any(abs(d["t"] - t) <= win for d in details)
    return round(hit / n, 3) if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="open/videos")
    ap.add_argument("--labels", default="open/open_labels.xlsx")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    settings = load_settings()
    models = build_models(settings)
    rules, facts = load_rules(APP_DIR / "rules.yaml"), load_facts(APP_DIR / "facts.yaml")
    labels = load_labels(Path(args.labels))
    types = {(r["video"], float(r["t"])): r["rule_id"] for r in csv.DictReader(open("data/types.csv"))}
    layouts = {r["video"]: r["layout"] for r in csv.DictReader(open("data/layouts.csv"))}
    only = set(filter(None, args.only.split(",")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    preds: dict[str, list[Violation]] = {}
    details: dict[str, list[dict]] = {}
    errors: list[str] = []
    for f in sorted(Path(args.videos).glob("*.mov")):
        name = video_key(f.name)
        if name in DUPLICATES or (only and name not in only):
            continue
        t0 = time.monotonic()
        try:
            verdict, violations, debug = analyze(f, models, settings, rules, facts)
        except Exception as e:
            print(f"{name} ERROR {type(e).__name__}: {e}", flush=True)
            preds[name] = []
            details[name] = []
            errors.append(name)
            continue
        preds[name] = violations
        details[name] = debug.get("candidates_detail", [])
        (out / f"{name}.json").write_text(json.dumps({"verdict": verdict, "debug": debug, "violations": [
            {k: v for k, v in x.model_dump().items() if k != "evidence_frames"} for x in violations]}, ensure_ascii=False, indent=1, default=str))
        print(f"{name:6} {layouts.get(name, '?')} {verdict:12} {round(time.monotonic() - t0):>4}s labels={labels.get(name)} "
              f"pred={[(v.rule_id, v.timecode_sec) for v in violations]}", flush=True)
    metrics = score(preds, labels, types, layouts)
    metrics["candidate_recall_3s"] = candidate_recall(details, labels)
    metrics["errors"] = errors
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
