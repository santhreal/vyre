#!/usr/bin/env python3
"""Behavior contracts for the documentation authority manifest."""

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
        (self.docs / "owner.md").write_text("# Owner\n", encoding="utf-8")
        DOCS_MANIFEST.DOCS = self.docs
        self.owners = {"docs": "owner.md"}

    def page(
        self,
        path: str,
        *,
        status: str = "current",
        audience: str = "user",
        owner: str = "docs",
        authority: str = "self",
        generation: str = "manual",
        generator: str | None = None,
        nav: bool | None = None,
    ) -> dict[str, object]:
        page: dict[str, object] = {
            "path": path,
            "title": path,
            "status": status,
            "audience": audience,
            "owner": owner,
            "kind": "guide",
            "section": "User workflows",
            "authority": authority,
            "generation": generation,
            "nav": status in {"current", "generated"} if nav is None else nav,
        }
        if generator is not None:
            page["generator"] = generator
        return page

    def test_valid_manifest_accepts_authority_and_generation_once(self) -> None:
        (self.docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
        (self.docs / "generated.md").write_text("# Generated\n", encoding="utf-8")
        (self.docs / "source.toml").write_text("version = 1\n", encoding="utf-8")
        (self.docs / "generator.py").write_text("# generator\n", encoding="utf-8")
        pages = [
            self.page("guide.md"),
            self.page(
                "generated.md",
                status="generated",
                audience="extension",
                authority="source.toml",
                generation="generated",
                generator="generator.py",
            ),
        ]

        self.assertEqual(
            DOCS_MANIFEST.validate(
                pages, {"guide.md", "generated.md"}, self.owners
            ),
            [],
        )

    def test_manifest_rejects_lifecycle_owner_and_provenance_drift(self) -> None:
        (self.docs / "declared.md").write_text("# Declared\n", encoding="utf-8")
        (self.docs / "unclassified.md").write_text("# Unclassified\n", encoding="utf-8")
        pages = [
            self.page("declared.md", status="archived", nav=True),
            self.page(
                "declared.md",
                status="generated",
                owner="removed-owner",
                authority="missing.toml",
                generation="generated",
                generator="missing.py",
            ),
            self.page("archive/old.md"),
        ]

        failures = DOCS_MANIFEST.validate(
            pages,
            {"declared.md", "unclassified.md", "archive/old.md"},
            self.owners,
        )

        for expected in (
            "duplicate DOCS.toml page",
            "unclassified documentation page",
            "inactive pages must set nav = false",
            "unknown or deleted documentation owner",
            "authority source does not exist",
            "generator does not exist",
            "historical directories require archived",
        ):
            self.assertTrue(
                any(expected in item for item in failures),
                f"Fix: missing negative fixture for {expected}: {failures}",
            )

    def test_external_documents_reject_internal_execution_process(self) -> None:
        (self.docs / "public.md").write_text(
            "# Public\n\nRead BACKLOG.md during Phase 3.\n", encoding="utf-8"
        )
        failures = DOCS_MANIFEST.validate(
            [self.page("public.md", audience="extension")],
            {"public.md"},
            self.owners,
        )

        self.assertTrue(any("leaks execution backlog" in item for item in failures))
        self.assertTrue(any("leaks internal phase identifier" in item for item in failures))

    def test_contributor_process_does_not_redefine_public_contract(self) -> None:
        (self.docs / "contributor.md").write_text(
            "# Contributor\n\nUpdate BACKLOG.md before Phase 3.\n", encoding="utf-8"
        )
        failures = DOCS_MANIFEST.validate(
            [self.page("contributor.md", audience="contributor")],
            {"contributor.md"},
            self.owners,
        )

        self.assertEqual(failures, [])

    def test_generated_page_requires_source_and_generator(self) -> None:
        (self.docs / "generated.md").write_text("# Generated\n", encoding="utf-8")
        failures = DOCS_MANIFEST.validate(
            [
                self.page(
                    "generated.md",
                    status="generated",
                    authority="owner.md",
                    generation="generated",
                )
            ],
            {"generated.md"},
            self.owners,
        )

        self.assertTrue(any("generated page must name one generator" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
