from __future__ import annotations

import runpy
from pathlib import Path
from urllib.error import HTTPError

import pytest

CHECK_TAG = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker/check_release_tag.py"))["check_tag"]


def version(tags: list[str]) -> dict:
    return {"metadata": {"container": {"tags": tags}}}


def test_existing_tag_on_later_page_is_rejected() -> None:
    calls = []

    def get(path: str):
        calls.append(path)
        if path.startswith("/repos/"):
            return {"owner": {"type": "Organization"}}
        if path.endswith("&page=1"):
            return [version([])] * 100
        return [version(["0.1.0-unified"])]

    with pytest.raises(RuntimeError, match="already exists"):
        CHECK_TAG("Owner/Repo", "0.1.0-unified", get)
    assert calls[-1] == "/orgs/Owner/packages/container/repo/versions?per_page=100&page=2"


def test_new_tag_in_existing_package_is_allowed() -> None:
    def get(path: str):
        if path.startswith("/repos/"):
            return {"owner": {"type": "User"}}
        return [version(["0.1.0-unified"])]

    CHECK_TAG("Owner/Repo", "0.1.1-unified", get)


@pytest.mark.parametrize("status", (401, 403, 404, 429, 500))
def test_only_first_package_404_is_allowed(status: int) -> None:
    def get(path: str):
        if path.startswith("/repos/"):
            return {"owner": {"type": "User"}}
        raise HTTPError(path, status, "request failed", {}, None)

    if status == 404:
        CHECK_TAG("Owner/Repo", "0.1.0-unified", get)
    else:
        with pytest.raises(HTTPError):
            CHECK_TAG("Owner/Repo", "0.1.0-unified", get)


def test_repository_access_error_is_never_treated_as_first_publication() -> None:
    def get(path: str):
        raise HTTPError(path, 404, "request failed", {}, None)

    with pytest.raises(HTTPError):
        CHECK_TAG("Owner/Repo", "0.1.0-unified", get)
