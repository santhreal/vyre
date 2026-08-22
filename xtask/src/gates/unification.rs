//! Cross-crate unifications stay unified.
//!
//! Each row names a refactor that already landed and counts the sites that would
//! undo it. A second exhaustive child match, a second fusion planning entry
//! point, a revived auto-inference helper: each is one owner turning back into
//! two. A row whose declared paths have moved is unmeasurable, which is worse
//! than a violation, so a missing path is itself a finding.

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{contains_word, Tree};

/// One tracked unification: what a regression looks like, where to look, and how
/// many sites the landed design leaves behind.
struct Row {
    name: &'static str,
    roots: &'static [&'static str],
    line: fn(&str) -> bool,
    ceiling: usize,
    /// Why the ceiling is what it is.
    rationale: &'static str,
}

/// The five tracked unifications.
///
/// Two rows carry a ceiling of one because their unification is achieved and the
/// single remaining site is the owner. The other three carry zero because the
/// shape they look for should not exist at all.
const ROWS: &[Row] = &[
    Row {
        name: "child-bodies-owner",
        roots: &["vyre-foundation/src"],
        line: |line| contains_word(line, "fn child_bodies"),
        ceiling: 1,
        rationale: "child enumeration has one public owner, so a second exhaustive child \
                    match is a duplicate",
    },
    Row {
        name: "buffer-access-auto-inference",
        roots: &[
            "vyre-foundation/src/lower",
            "vyre-driver-wgpu/src",
            "vyre-runtime/src/resident_work_queue",
        ],
        line: |line| {
            line.contains("BufferAccess::infer")
                || line.contains("BufferAccess::auto")
                || line.contains("BufferAccess::derive_from")
        },
        ceiling: 0,
        rationale: "buffer access is declared, never inferred behind the caller's back",
    },
    Row {
        name: "cpu-reference-implementations",
        roots: &["vyre-foundation/src", "vyre-reference/src"],
        line: |line| contains_word(line, "fn cpu_reference"),
        ceiling: 0,
        rationale: "a reference implementation lives with the operation it checks, not in a \
                    parallel tree",
    },
    Row {
        name: "fusion-planning-entry",
        roots: &[
            "vyre-foundation/src/execution_plan",
            "vyre-pass-engine/src",
            "vyre-runtime/src/resident_work_queue",
        ],
        line: |line| {
            contains_word(line, "fn plan_fusion")
                || contains_word(line, "fn fuse_programs")
                || contains_word(line, "fn tensor_network_fusion_order")
        },
        ceiling: 1,
        rationale: "there is one fusion planning entry point and it is the owner",
    },
    Row {
        name: "pipeline-cache-in-backend",
        roots: &["vyre-driver-wgpu/src"],
        line: |line| line.contains("impl PipelineCacheStore for"),
        ceiling: 0,
        rationale: "the pipeline cache substrate lives in the driver tier, not in a backend \
                    crate",
    },
];

/// Landed cross-crate unifications stay landed.
pub struct Unification;

impl crate::gate::GateBehavior for Unification {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("unification source files", tree.all_rust().len());
        for row in ROWS {
            let mut missing = false;
            for root in row.roots {
                if !tree.exists(root) {
                    missing = true;
                    report.find(Finding::in_file(
                        *root,
                        format!("{} scans a path that does not exist", row.name),
                        "repoint the row at the path the code moved to; a row over a missing \
                         path measures nothing and passes forever",
                    ));
                }
            }
            if missing {
                continue;
            }
            let files = tree
                .rust(row.roots)?
                .into_iter()
                .filter(|path| {
                    let text = path.to_string_lossy();
                    !(text.contains("/tests/")
                        || text.ends_with("_tests.rs")
                        || text.contains("test_fixtures"))
                })
                .collect::<Vec<_>>();
            let hits = tree.hits(&files, |line| (row.line)(line))?;
            report.note(format!(
                "{}: {} site(s), ceiling {}",
                row.name,
                hits.len(),
                row.ceiling
            ));
            let count = hits.len();
            if count <= row.ceiling {
                continue;
            }
            for hit in hits {
                report.find(Finding::at(
                    hit.file,
                    hit.line,
                    format!(
                        "{} has {count} site(s) against a ceiling of {}: {}",
                        row.name, row.ceiling, hit.text
                    ),
                    format!("{}; fold the extra site into the owner", row.rationale),
                ));
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::gate::GateBehavior;
    use crate::gates::fixture_checkout;

    /// A git checkout carrying every root the rows scan, each holding one file.
    ///
    /// The roots have to exist, because a row over a missing path is itself a
    /// finding, which is the whole point of the second fixture below.
    fn checkout() -> (TempDir, PathBuf) {
        let roots: Vec<&str> = ROWS
            .iter()
            .flat_map(|row| row.roots.iter().copied())
            .collect();
        fixture_checkout::checkout_with_roots(&roots)
    }

    /// Every finding message, joined.
    fn messages(report: &Report) -> String {
        report.finding_messages()
    }

    /// WHY: this gate is a ratchet, and a ratchet is only worth its run time if
    /// the direction it forbids is observed to be red. Two sites of a surface
    /// whose ceiling is one is exactly the regression each row names, and the
    /// report has to carry the count and the ceiling or the reader cannot tell a
    /// legitimate owner from a duplicate.
    #[test]
    fn a_second_owner_of_a_unified_surface_is_reported() {
        let (_temporary, root) = checkout();
        fs::write(
            root.join("vyre-foundation/src/first.rs"),
            "pub fn child_bodies() {}\n",
        )
        .expect("the owner");
        fs::write(
            root.join("vyre-foundation/src/second.rs"),
            "pub fn child_bodies() {}\n",
        )
        .expect("the duplicate");

        let report = Unification
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        let reported = messages(&report);
        assert!(
            reported.contains("child-bodies-owner has 2 site(s) against a ceiling of 1"),
            "the duplicate is reported with its count and ceiling: {reported}"
        );
    }

    /// WHY: three of the five rows in the shell version of this ratchet scanned
    /// paths the code had moved out of and scored zero, which is at or below every
    /// ceiling, so they passed by measuring nothing. A row over a missing path is
    /// therefore a finding in its own right, and it must name the path rather than
    /// report a clean count.
    #[test]
    fn a_row_that_scans_a_missing_path_is_reported() {
        let (_temporary, root) = checkout();
        fs::remove_dir_all(root.join("vyre-driver-wgpu/src")).expect("a root the code left");

        let report = Unification
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate runs");
        let reported = messages(&report);
        assert!(
            reported.contains("scans a path that does not exist"),
            "the unmeasurable row is reported: {reported}"
        );
        assert!(
            !report
                .notes
                .iter()
                .any(|note| note.contains("pipeline-cache-in-backend")),
            "and it is not also counted as clean: {:?}",
            report.notes
        );
    }
}
