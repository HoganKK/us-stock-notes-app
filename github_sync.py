from __future__ import annotations

import base64
from typing import Any

import requests


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_file_content(
    token: str,
    repo: str,
    path: str,
    branch: str = "main",
) -> tuple[str, str | None]:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=30)
    if resp.status_code == 404:
        return "", None
    resp.raise_for_status()
    obj = resp.json()
    content_b64 = obj.get("content", "") or ""
    sha = obj.get("sha")
    text = base64.b64decode(content_b64).decode("utf-8")
    return text, sha


def upsert_file_content(
    token: str,
    repo: str,
    path: str,
    branch: str,
    text: str,
    commit_message: str,
) -> dict[str, Any]:
    current_text, current_sha = get_file_content(token=token, repo=repo, path=path, branch=branch)
    encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    payload: dict[str, Any] = {
        "message": commit_message,
        "content": encoded,
        "branch": branch,
    }
    if current_sha:
        payload["sha"] = current_sha
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    resp = requests.put(url, headers=_headers(token), json=payload, timeout=35)
    resp.raise_for_status()
    return resp.json()

