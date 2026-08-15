//! Production source files stay under a per-file line cap.
//!
//! Two ratchets and two caps. A file under a core crate carries the cap it was
//! measured at, plus five percent, so a typo fix lands and a two hundred line
//! addition does not. Everything else gets a flat cap. Test trees, benches, fuzz
//! targets and the task runners get a looser one because a grammar table or a
//! catalog dump is legitimately long.

use std::path::PathBuf;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Flat cap for a production file outside the core crates.
const MAX_LINES: usize = 3000;
/// Flat cap for a production file under a core crate with no measured row.
const CORE_MAX_LINES: usize = 2500;
/// Cap for tests, benches, fuzz targets, the conform tree and the task runners.
const TEST_MAX_LINES: usize = 8000;

/// Core crates whose files carry measured caps.
const CORE_ROOTS: &[&str] = &[
    "vyre-foundation/src/",
    "vyre-runtime/src/",
    "vyre-reference/src/",
    "vyre-driver-wgpu/src/",
    "vyre-libs/src/",
    "vyre-primitives/src/",
];

/// Measured line counts for files under the core crates. The cap is this number
/// plus five percent, rounded up.
const CORE_MEASURED: &[(&str, usize)] = &[
    ("vyre-libs/src/parsing/c/preprocess/mod.rs", 981),
    ("vyre-driver-wgpu/src/backend_impl.rs", 1360),
    ("vyre-driver-wgpu/src/pipeline/mod.rs", 900),
    ("vyre-libs/src/parsing/c/parse/structure/mod.rs", 844),
    ("vyre-driver-wgpu/src/buffer/handle/mod.rs", 674),
    ("vyre-driver-wgpu/src/buffer/bind_group_cache/mod.rs", 331),
    ("vyre-driver-wgpu/src/engine/record_and_readback/mod.rs", 829),
    ("vyre-reference/src/workgroup.rs", 810),
    ("vyre-driver-wgpu/src/engine/multi_gpu/mod.rs", 808),
    ("vyre-foundation/src/optimizer/fact_cache/mod.rs", 570),
    ("vyre-runtime/src/tenant/handle.rs", 443),
    ("vyre-runtime/src/tenant/registry.rs", 302),
    ("vyre-foundation/src/transform/visit/mod.rs", 789),
    ("vyre-foundation/src/optimizer/rewrite.rs", 754),
    ("vyre-driver-wgpu/src/buffer/pool.rs", 910),
    ("vyre-runtime/src/uring/stream.rs", 830),
    ("vyre-foundation/src/ir_inner/model/program/meta/mod.rs", 940),
    ("vyre-libs/src/parsing/c/sema/registry/mod.rs", 660),
    ("vyre-libs/src/parsing/c/lower/semantic_edges/mod.rs", 650),
    ("vyre-foundation/src/validate/expr_rules.rs", 646),
    (
        "vyre-foundation/src/ir_inner/model/program/buffer_decl/mod.rs",
        725,
    ),
    ("vyre-foundation/src/serial/wire/decode/from_wire/mod.rs", 606),
    ("vyre-foundation/src/transform/autodiff/grad/mod.rs", 702),
    ("vyre-libs/src/nn/attention/softmax.rs", 592),
    ("vyre-libs/src/parsing/python/parse/structure.rs", 700),
    ("vyre-foundation/src/validate/typecheck/mod.rs", 578),
    ("vyre-libs/src/math/linalg/matmul.rs", 815),
    ("vyre-driver-wgpu/src/runtime/readback_ring/ring.rs", 459),
    ("vyre-libs/src/decode/inflate.rs", 554),
    ("vyre-foundation/src/serial/wire/encode/to_wire/mod.rs", 690),
    ("vyre-libs/src/parsing/python/lex.rs", 660),
    ("vyre-runtime/src/replay/mod.rs", 549),
    ("vyre-primitives/src/matching/region.rs", 544),
    ("vyre-primitives/src/matching/dfa_compile/compile.rs", 344),
    ("vyre-libs/src/parsing/go/parse/structure.rs", 539),
    ("vyre-foundation/src/ir_inner/model/expr/mod.rs", 539),
    ("vyre-foundation/src/optimizer/mod.rs", 970),
    ("vyre-foundation/src/execution_plan/mod.rs", 740),
    ("vyre-runtime/src/uring/ring.rs", 685),
    ("vyre-foundation/src/execution_plan/policy.rs", 660),
    ("vyre-libs/src/parsing/c/sema/lookup/mod.rs", 507),
    ("vyre-primitives/src/math/semiring_gemm/mod.rs", 535),
];

/// Per-file ceilings for files the split audit tracks outside the core crates.
/// A row naming a core-crate path is shadowed by the core ceiling above, which is
/// the tighter of the two, and is reported so the row can be retired.
const AUDIT_CEILINGS: &[(&str, usize)] = &[("vyre-driver-cuda/src/codegen/mod.rs", 1160)];

/// Production source files stay under their cap.
pub struct FileSize;

impl Gate for FileSize {
    fn name(&self) -> &'static str {
        "file-size"
    }

    fn help(&self) -> &'static str {
        "source files over their per-file line cap, and ratchet rows that name nothing"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let mut rows: Vec<(usize, usize, PathBuf)> = Vec::new();
        for path in tree.paths() {
            let relative = path.to_string_lossy();
            if !relative.ends_with(".rs") || !relative.contains("/src/") {
                continue;
            }
            let lines = tree.read(path)?.matches('\n').count();
            let cap = cap_for(&relative);
            if lines > cap {
                report.find(Finding::in_file(
                    path.clone(),
                    format!("{lines} lines exceeds the cap of {cap}"),
                    "split the file into focused modules; if the size is structural, raise \
                     the measured row in this gate with the rationale in the same change",
                ));
            }
            rows.push((lines, cap, path.clone()));
        }
        report.note(format!("{} source files scanned", rows.len()));
        if ctx.has("--report") {
            rows.sort_by(|left, right| right.0.cmp(&left.0).then(left.2.cmp(&right.2)));
            for (lines, cap, path) in rows {
                report.note(format!("{lines} {cap} {}", path.display()));
            }
        }

        for (table, rows) in [("core", CORE_MEASURED), ("audit", AUDIT_CEILINGS)] {
            for (path, _) in rows {
                if !tree.exists(path) {
                    report.find(Finding::in_file(
                        *path,
                        format!("the {table} ratchet names a file that does not exist"),
                        "delete the row; a ratchet on a missing file holds no ceiling",
                    ));
                }
            }
        }

        for (path, ceiling) in AUDIT_CEILINGS {
            if is_core(path) {
                report.find(Finding::in_file(
                    *path,
                    format!(
                        "the audit ceiling of {ceiling} is shadowed by the core ratchet, which \
                         is the tighter of the two"
                    ),
                    "delete the audit row; the core ratchet is the live ceiling for this file",
                ));
            }
        }

        Ok(report)
    }
}

/// The cap a path is held to.
fn cap_for(path: &str) -> usize {
    if is_outside_production(path) {
        return TEST_MAX_LINES;
    }
    if is_core(path) {
        return match measured(CORE_MEASURED, path) {
            Some(base) => base + base.div_ceil(20),
            None => CORE_MAX_LINES,
        };
    }
    match measured(AUDIT_CEILINGS, path) {
        Some(base) => base + base.div_ceil(20),
        None => MAX_LINES,
    }
}

fn measured(rows: &[(&str, usize)], path: &str) -> Option<usize> {
    rows.iter()
        .find(|(candidate, _)| *candidate == path)
        .map(|(_, value)| *value)
}

/// Test trees, benches, fuzz targets, the conform tree and the task runners.
fn is_outside_production(path: &str) -> bool {
    path.contains("/tests/")
        || path.contains("/benches/")
        || path.contains("/fuzz/")
        || path.ends_with("tests.rs")
        || path.contains("xtask/src/")
        || path.contains("xtask-registry/src/")
        || path.contains("xtask-evidence/src/")
        || path.starts_with("conform/")
}

fn is_core(path: &str) -> bool {
    CORE_ROOTS.iter().any(|root| path.starts_with(root))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the five percent headroom is what lets a typo fix land. It rounds up,
    /// so a small file still gets at least one line of slack, and a cap that
    /// rounded down would make the gate fire on formatting.
    #[test]
    fn headroom_rounds_up() {
        assert_eq!(cap_for("vyre-libs/src/decode/inflate.rs"), 554 + 28);
        assert_eq!(cap_for("vyre-driver-cuda/src/codegen/mod.rs"), 1160 + 58);
    }

    /// WHY: the core ratchet has to win over the audit ceiling, because it is the
    /// tighter number. A file in both tables that took the audit ceiling would
    /// silently gain hundreds of lines of room.
    #[test]
    fn the_core_ratchet_wins_over_the_audit_ceiling() {
        let path = "vyre-foundation/src/transform/visit/mod.rs";
        assert!(is_core(path));
        assert_eq!(cap_for(path), 789 + 40);
    }

    /// WHY: the resident work queue left the exclusion when the runtime
    /// restructure finished. Every file under it is now core, takes the core
    /// cap, and carries no measured row because every one of them is under it.
    #[test]
    fn the_resident_work_queue_tree_is_core() {
        let path = "vyre-runtime/src/resident_work_queue/scheduler/mod.rs";

        assert!(is_core(path));
        assert_eq!(cap_for(path), CORE_MAX_LINES);
    }

    /// WHY: the looser cap exists for generated and harness code. A file under a
    /// core crate's tests directory must take it, or extracting tests out of an
    /// oversized file would trade one violation for another.
    #[test]
    fn harness_trees_take_the_looser_cap() {
        assert_eq!(
            cap_for("vyre-foundation/src/optimizer/passes/fusion_tests.rs"),
            TEST_MAX_LINES
        );
        assert_eq!(cap_for("xtask/src/gates/hygiene_matrix.rs"), TEST_MAX_LINES);
        assert_eq!(cap_for("conform/vyre-conform/src/lib.rs"), TEST_MAX_LINES);
    }
}
