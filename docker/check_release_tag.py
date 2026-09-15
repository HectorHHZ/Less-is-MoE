"""Reject existing GHCR version tags, including on a workflow rerun."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def github_get(path: str):
    request = Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def check_tag(repository: str, tag: str, get=github_get) -> None:
    owner, name = repository.split("/")
    # This also verifies authentication before interpreting a package 404.
    repo = get(f"/repos/{quote(owner)}/{quote(name)}")
    owner_kind = "orgs" if repo["owner"]["type"] == "Organization" else "users"
    base = f"/{owner_kind}/{quote(owner)}/packages/container/{quote(name.lower(), safe='')}"
    page = 1
    while True:
        try:
            versions = get(f"{base}/versions?per_page=100&page={page}")
        except HTTPError as error:
            if error.code == 404 and page == 1:
                # GHCR's manifest endpoint can return DENIED before a package's
                # first push. The authenticated Packages API represents this as
                # 404. A token with packages:write also needs package read access.
                print("No accessible existing package; this is the first publication or package access must be granted.")
                return
            raise
        for version in versions:
            if tag in version["metadata"]["container"]["tags"]:
                raise RuntimeError(f"Tag {tag} already exists; publish a new version instead")
        if len(versions) < 100:
            print(f"Version tag {tag} is available")
            return
        page += 1


if __name__ == "__main__":
    check_tag(os.environ["GITHUB_REPOSITORY"], f"{os.environ['VERSION']}-unified")
