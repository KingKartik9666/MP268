from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "Dockerfile": ROOT / "Dockerfile",
    "Kubernetes Deployment": ROOT / "k8s" / "deployment.yaml",
    "Kubernetes Service": ROOT / "k8s" / "service.yaml",
}


def read_file(label: str, path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def check(condition: bool, message: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return condition


def extract_label_block(text: str, heading: str):
    """Extract simple indented YAML key/value pairs beneath a heading."""
    match = re.search(
        rf"(?m)^{re.escape(heading)}:\s*\n((?:^[ \t]+[A-Za-z0-9_.-]+:\s*.+\n?)*)",
        text,
    )
    if not match:
        return {}
    return dict(
        re.findall(
            r"(?m)^[ \t]+([A-Za-z0-9_.-]+):\s*['\"]?([^'\"\s#]+)",
            match.group(1),
        )
    )
def main() -> int:
    print("MP268 deployment preflight")
    print("=" * 30)

    files = {}
    all_ok = True

    for label, path in REQUIRED_FILES.items():
        try:
            files[label] = read_file(label, path)
            all_ok &= check(True, f"{label} exists")
        except FileNotFoundError as exc:
            all_ok &= check(False, str(exc))

    if len(files) != len(REQUIRED_FILES):
        print("\nPreflight failed: one or more required files are missing.")
        return 1

    dockerfile = files["Dockerfile"]
    deployment = files["Kubernetes Deployment"]
    service = files["Kubernetes Service"]

    all_ok &= check(
        bool(re.search(r"(?im)^\s*FROM\s+python:3(?:\.\d+)?-slim\b", dockerfile)),
        "Dockerfile uses a slim Python 3 base image",
    )
    all_ok &= check(
        "requirements.txt" in dockerfile
        and bool(re.search(r"(?im)^\s*RUN\s+.*pip\s+install", dockerfile)),
        "Dockerfile installs Python requirements",
    )
    all_ok &= check(
        bool(re.search(r"(?im)^\s*EXPOSE\s+8000\b", dockerfile)),
        "Dockerfile exposes API port 8000",
    )
    all_ok &= check(
        bool(re.search(r"(?m)^\s*replicas:\s*2\s*(?:#.*)?$", deployment)),
        "Kubernetes Deployment requests two replicas",
    )
    all_ok &= check(
        bool(re.search(r"(?m)^\s*image:\s*mp268-api:1\.0\s*(?:#.*)?$", deployment)),
        "Kubernetes Deployment uses mp268-api:1.0",
    )
    all_ok &= check(
        "livenessProbe:" in deployment
        and "readinessProbe:" in deployment
        and "/health" in deployment,
        "Deployment defines liveness and readiness health probes",
    )
    all_ok &= check(
        bool(re.search(r"(?m)^\s*port:\s*80\s*(?:#.*)?$", service))
        and bool(re.search(r"(?m)^\s*targetPort:\s*8000\s*(?:#.*)?$", service)),
        "Service maps port 80 to container port 8000",
    )

    pod_labels = extract_label_block(deployment, "labels")
    service_selector = extract_label_block(service, "selector")
    selector_matches = bool(service_selector) and all(
        pod_labels.get(key) == value for key, value in service_selector.items()
    )
    all_ok &= check(
        selector_matches,
        "Service selector matches Deployment pod labels",
    )

    print("\n" + ("Preflight passed." if all_ok else "Preflight failed."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
