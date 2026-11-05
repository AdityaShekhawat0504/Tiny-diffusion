"""Repo hygiene checks that need no torch."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_images_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", readme)
    assert images, "README shows no images"
    missing = [p for p in images if not (ROOT / p).is_file()]
    assert not missing, f"README links missing images: {missing}"


def test_requirements_are_pinned():
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (ROOT / name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            assert "==" in line, f"{name}: unpinned requirement {line!r}"
