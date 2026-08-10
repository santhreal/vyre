#!/usr/bin/env python3
"""Behavior contracts for the documentation lifecycle manifest."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("docs_manifest.py")
SPEC = importlib.util.spec_from_file_location("docs_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DOCS_MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCS_MANIFEST)


class DocumentationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.docs = Path(self.temp.name) / "docs"
        self.docs.mkdir()
        DOCS_MANIFEST.DOCS = self.docs

    def page(
        self,
        path: str,
        *,
        status: str = "current",
        source: str | None = None,
        nav: bool | None = None,
    ) -> dict[str, object]:
        return {
            "path": path,
            "title": path,
            "status": status,
            "class": "guide",
            "section": "Guide",
            "source": source if source is not None else path,
            "nav": status in {"current", "generated"} if nav is None else nav,
        }

    def test_valid_manifest_accepts_each_document_once(self) -> None:
        (self.docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
        (self.docs / "generated.md").write_text("# Generated\n", encoding="utf-8")
        (self.docs / "source.toml").write_text("version = 1\n", encoding="utf-8")
        pages = [
            self.page("guide.md"),
            self.page("generated.md", status="generated", source="source.toml"),
        ]

        self.assertEqual(
            DOCS_MANIFEST.validate(pages, {"guide.md", "generated.md"}), []
        )

    def test_manifest_rejects_lifecycle_and_provenance_drift(self) -> None:
        (self.docs / "declared.md").write_text("# Declared\n", encoding="utf-8")
        (self.docs / "unclassified.md").write_text("# Unclassified\n", encoding="utf-8")
        pages = [
            self.page("declared.md", status="archived", nav=True),
            self.page("declared.md", status="generated", source="missing.toml"),
            self.page("archive/old.md"),
        ]

        failures = DOCS_MANIFEST.validate(
            pages, {"declared.md", "unclassified.md", "archive/old.md"}
        )

        self.assertTrue(any("duplicate DOCS.toml page" in item for item in failures))
        self.assertTrue(any("unclassified documentation page" in item for item in failures))
        self.assertTrue(any("inactive pages must set nav = false" in item for item in failures))
        self.assertTrue(any("generated source does not exist" in item for item in failures))
        self.assertTrue(any("historical directories require archived" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
