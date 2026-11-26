"""MP268 deployment configuration tests.

Upload this single file to: tests/test_deployment_config.py

Run from the repository root:
    pytest -q tests/test_deployment_config.py

Uses only Python's standard library plus pytest. It checks the existing
Dockerfile and Kubernetes YAML files without requiring PyYAML.
"""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT = ROOT / "k8s" / "deployment.yaml"
SERVICE = ROOT / "k8s" / "service.yaml"


def read_required_file(path: Path) -> str:
    """Read a required deployment file with a useful test failure."""
    assert path.is_file(), f"Required deployment file is missing: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerfile_text():
    return read_required_file(DOCKERFILE)


@pytest.fixture(scope="module")
def deployment_text():
    return read_required_file(DEPLOYMENT)


@pytest.fixture(scope="module")
def service_text():
    return read_required_file(SERVICE)


def test_dockerfile_uses_python_base_image(dockerfile_text):
    assert re.search(r"(?im)^\s*FROM\s+python:3(?:\.\d+)?-slim\b", dockerfile_text), (
        "Dockerfile should use a slim Python 3 base image."
    )


def test_dockerfile_installs_requirements(dockerfile_text):
    assert "requirements.txt" in dockerfile_text
    assert re.search(r"(?im)^\s*RUN\s+.*pip\s+install", dockerfile_text), (
        "Dockerfile should install the pinned Python dependencies."
    )


def test_dockerfile_exposes_api_port(dockerfile_text):
    assert re.search(r"(?im)^\s*EXPOSE\s+8000\b", dockerfile_text), (
        "Dockerfile should expose port 8000 for the API."
    )


def test_kubernetes_deployment_has_two_replicas(deployment_text):
    assert re.search(r"(?m)^\s*replicas:\s*2\s*(?:#.*)?$", deployment_text), (
        "Deployment should request two replicas."
    )


def test_kubernetes_deployment_uses_expected_image(deployment_text):
    assert re.search(r"(?m)^\s*image:\s*mp268-api:1\.0\s*(?:#.*)?$", deployment_text), (
        "Deployment should use image mp268-api:1.0."
    )


def test_kubernetes_deployment_has_health_probes(deployment_text):
    assert re.search(r"(?m)^\s*livenessProbe:\s*$", deployment_text)
    assert re.search(r"(?m)^\s*readinessProbe:\s*$", deployment_text)
    assert "/health" in deployment_text


def test_kubernetes_service_exposes_port_80_to_8000(service_text):
    assert re.search(r"(?m)^\s*port:\s*80\s*(?:#.*)?$", service_text)
    assert re.search(r"(?m)^\s*targetPort:\s*8000\s*(?:#.*)?$", service_text)


def test_service_selector_matches_deployment_pod_label(deployment_text, service_text):
    """Ensure the Service routes traffic to the Deployment's app label."""
    deployment_match = re.search(
        r"(?ms)^\s*template:\s*\n.*?^\s*labels:\s*\n(?P<labels>(?:^\s+\w[\w.-]*:\s*.+\n?)+)",
        deployment_text,
    )
    service_match = re.search(
        r"(?ms)^\s*selector:\s*\n(?P<labels>(?:^\s+\w[\w.-]*:\s*.+\n?)+)",
        service_text,
    )

    assert deployment_match, "Could not find pod template labels in deployment.yaml."
    assert service_match, "Could not find Service selector in service.yaml."

    deployment_labels = dict(
        re.findall(r"(?m)^\s+([\w.-]+):\s*['\"]?([^'\"\s#]+)", deployment_match.group("labels"))
    )
    service_selector = dict(
        re.findall(r"(?m)^\s+([\w.-]+):\s*['\"]?([^'\"\s#]+)", service_match.group("labels"))
    )

    assert service_selector, "Service selector must not be empty."
    for key, value in service_selector.items():
        assert deployment_labels.get(key) == value, (
            f"Service selector {key}={value} does not match Deployment pod labels."
        )
