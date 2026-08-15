//! Cross-crate unifications stay unified.
//!
//! Each row names a refactor that already landed and counts the sites that would
//! undo it. A second exhaustive child match, a second fusion planning entry
//! point, a revived auto-inference helper: each is one owner turning back into
//! two. A row whose declared paths have moved is unmeasurable, which is worse
//! than a violation, so a missing path is itself a finding.

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
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
            "vyre-runtime/src/megakernel",
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
            "vyre-runtime/src/megakernel",
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

impl Gate for Unification {
    fn name(&self) -> &'static str {
        "unification"
    }

    fn help(&self) -> &'static str {
        "sites that would turn one owner of a unified surface back into two"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
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
