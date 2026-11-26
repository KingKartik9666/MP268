"""Fill reports/findings.yaml (the marking form) from the code's own outputs. Owner: Kartik.

Run from the repo root, after findings / train final / monitor:
    python -m mp268.report
Sources: mp268.findings.run_all() (data_issues), reports/v1.json (tracking),
reports/monitoring.json (monitoring). Nothing is typed in by hand.
"""
from __future__ import annotations

import json

import yaml

from mp268.config import ROOT, load_params, resolve
from mp268.findings import run_all

HEADER = ("# MP268 findings --- filled by `python -m mp268.report` from the code outputs.\n"
          "# Data problems: src/mp268/findings.py, tracking: src/mp268/train.py, monitoring: src/mp268/monitor.py\n\n")


def build(params: dict) -> dict:
    v1 = json.loads(resolve("reports/v1.json").read_text())
    mon = json.loads(resolve("reports/monitoring.json").read_text())
    issues = run_all(params)
    todo = [k for k, v in issues.items() if not isinstance(v, dict)]
    if todo:
        raise SystemExit(f"data checks still TODO, fix them first: {todo}")
    return {
        "group_id": "G252",
        "problem_id": "MP268",
        "data_issues": issues,
        "tracking": {
            "experiment_name": v1["experiment_name"],
            "best_run_id": v1["best_run_id"],
            "registered_model_name": v1["registered_model_name"],
            "champion_version": mon["champion_version"],
        },
        "monitoring": {
            "psi": mon["psi"],
            "most_drifted_feature": mon["most_drifted_feature"],
            "mae_by_group": {g: mon["mae_by_group"][g] for g in ("Highway", "City", "Industrial")},
            "worst_served_group": mon["worst_served_group"],
            "mae_v1_on_check": mon["mae_v1_on_check"],
            "mae_v2_on_check": mon["mae_v2_on_check"],
            "decision": mon["decision"],
        },
    }


def check_against_template(filled: dict) -> None:
    """Every key of the faculty's template must be present (the marking tool reads these keys)."""
    template = ROOT / "findings.yaml"
    if not template.exists():
        return

    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + str(k))
            if isinstance(v, dict):
                out |= keys(v, prefix + str(k) + ".")
        return out

    missing = keys(yaml.safe_load(template.read_text(encoding="utf-8"))) - keys(filled)
    if missing:
        raise SystemExit(f"keys missing compared with the template: {sorted(missing)}")


def main() -> None:
    filled = build(load_params())
    check_against_template(filled)
    path = resolve("reports/findings.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump(filled, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
