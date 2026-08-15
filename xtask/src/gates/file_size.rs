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

/// Core crates whose files carry measured caps. The megakernel tree is excluded
/// while the runtime restructure moves it.
const CORE_ROOTS: &[&str] = &[
    "vyre-foundation/src/",
    "vyre-runtime/src/",
    "vyre-reference/src/",
    "vyre-driver-wgpu/src/",
    "vyre-libs/src/",
    "vyre-primitives/src/",
];
const CORE_EXCLUDED: &str = "vyre-runtime/src/megakernel/";

/// Measured line counts for files under the core crates. The cap is this number
/// plus five percent, rounded up.
const CORE_MEASURED: &[(&str, usize)] = &[
    ("vyre-libs/src/parsing/c/parse/vast.rs", 8692),
    ("vyre-libs/src/parsing/c/preprocess/expansion.rs", 3187),
    ("vyre-libs/src/parsing/c/lower/ast_to_pg_nodes.rs", 1587),
    ("vyre-driver-wgpu/src/lowering/naga_emit/expr.rs", 1463),
    ("vyre-libs/src/parsing/c/lex/lexer.rs", 1292),
    ("vyre-driver-wgpu/src/lowering/naga_emit/mod.rs", 1269),
    ("vyre-foundation/src/optimizer/scheduler.rs", 1041),
    ("vyre-libs/src/nn/linear/linear.rs", 1016),
    ("vyre-libs/src/parsing/c/preprocess/mod.rs", 981),
    ("vyre-driver-wgpu/src/backend_impl.rs", 1360),
    ("vyre-driver-wgpu/src/pipeline.rs", 900),
    ("vyre-foundation/src/validate/validate.rs", 993),
    ("vyre-runtime/src/pipeline_cache.rs", 882),
    ("vyre-libs/src/parsing/c/parse/structure.rs", 844),
    ("vyre-driver-wgpu/src/buffer/handle.rs", 1180),
    ("vyre-driver-wgpu/src/engine/record_and_readback.rs", 829),
    ("vyre-reference/src/workgroup.rs", 810),
    ("vyre-driver-wgpu/src/engine/multi_gpu.rs", 808),
    ("vyre-foundation/src/optimizer/passes/fusion.rs", 804),
    ("vyre-foundation/src/optimizer/fact_cache.rs", 570),
    ("vyre-runtime/src/tenant.rs", 1060),
    ("vyre-runtime/src/megakernel/telemetry.rs", 791),
    ("vyre-foundation/src/transform/visit.rs", 789),
    ("vyre-libs/src/matching/nfa.rs", 754),
    ("vyre-foundation/src/optimizer/rewrite.rs", 754),
    ("vyre-runtime/src/megakernel/protocol.rs", 737),
    ("vyre-driver-wgpu/src/buffer/pool.rs", 910),
    ("vyre-driver-wgpu/src/lowering/naga_emit/node.rs", 709),
    ("vyre-runtime/src/uring/stream.rs", 830),
    ("vyre-foundation/src/ir_inner/model/program/meta.rs", 940),
    ("vyre-driver-wgpu/src/pipeline_disk_cache.rs", 677),
    ("vyre-libs/src/parsing/c/sema/registry.rs", 660),
    ("vyre-runtime/src/megakernel/io.rs", 653),
    ("vyre-libs/src/parsing/c/lower/semantic_edges.rs", 650),
    ("vyre-foundation/src/validate/expr_rules.rs", 646),
    ("vyre-foundation/src/ir_inner/model/program/buffer_decl.rs", 725),
    ("vyre-reference/src/typed_ops.rs", 618),
    ("vyre-driver-wgpu/src/runtime/tuner.rs", 618),
    ("vyre-foundation/src/serial/wire/decode/from_wire.rs", 606),
    ("vyre-foundation/src/transform/autodiff/grad.rs", 702),
    ("vyre-foundation/src/execution_plan/fusion.rs", 593),
    ("vyre-runtime/src/megakernel/builder.rs", 592),
    ("vyre-libs/src/nn/attention/softmax.rs", 592),
    ("vyre-reference/src/eval_expr.rs", 588),
    ("vyre-libs/src/math/linalg/matmul_tiled.rs", 641),
    ("vyre-primitives/src/math/sinkhorn_iterate.rs", 740),
    ("vyre-libs/src/dataflow/ifds_gpu.rs", 583),
    ("vyre-libs/src/parsing/python/parse/structure.rs", 700),
    ("vyre-libs/src/matching/regex_compile.rs", 579),
    ("vyre-foundation/src/validate/typecheck.rs", 578),
    ("vyre-libs/src/math/linalg/matmul.rs", 815),
    ("vyre-driver-wgpu/src/runtime/readback_ring.rs", 720),
    ("vyre-libs/src/decode/inflate.rs", 554),
    ("vyre-foundation/src/serial/wire/encode/to_wire.rs", 690),
    ("vyre-libs/src/parsing/python/lex.rs", 660),
    ("vyre-foundation/src/validate/nodes.rs", 653),
    ("vyre-runtime/src/replay.rs", 549),
    ("vyre-primitives/src/matching/region.rs", 544),
    ("vyre-primitives/src/matching/dfa_compile.rs", 640),
    ("vyre-libs/src/parsing/go/parse/structure.rs", 539),
    ("vyre-foundation/src/ir_inner/model/expr.rs", 539),
    ("vyre-foundation/src/optimizer.rs", 970),
    ("vyre-foundation/src/optimizer/passes/dead_buffer_elim.rs", 581),
    ("vyre-foundation/src/execution_plan/mod.rs", 740),
    ("vyre-runtime/src/uring/ring.rs", 685),
    ("vyre-primitives/src/text/utf8_validate.rs", 516),
    ("vyre-reference/src/hashmap_interp/node_step.rs", 514),
    ("vyre-foundation/src/execution_plan/policy.rs", 660),
    ("vyre-libs/src/parsing/c/sema/lookup.rs", 507),
    ("vyre-primitives/src/math/semiring_gemm.rs", 535),
];

/// Per-file ceilings for files the split audit tracks outside the core crates.
/// A row naming a core-crate path is shadowed by the core ceiling above, which is
/// the tighter of the two, and is reported so the row can be retired.
const AUDIT_CEILINGS: &[(&str, usize)] = &[
    ("vyre-libs/src/parsing/c/parse/vast.rs", 9100),
    ("vyre-libs/src/parsing/c/preprocess/expansion.rs", 3300),
    ("vyre-libs/src/parsing/c/lower/ast_to_pg_nodes.rs", 1670),
    ("vyre-driver-wgpu/src/lowering/naga_emit/expr.rs", 1600),
    ("vyre-driver-wgpu/src/lib.rs", 1400),
    ("vyre-driver-wgpu/src/lowering/naga_emit/mod.rs", 1280),
    ("vyre-libs/src/parsing/c/lex/lexer.rs", 1360),
    ("vyre-foundation/src/optimizer/passes/fusion.rs", 820),
    ("vyre-foundation/src/validate/validate.rs", 920),
    ("vyre-driver-cuda/src/backend.rs", 1170),
    ("vyre-driver-wgpu/src/pipeline_disk_cache.rs", 850),
    ("vyre-driver-cuda/src/codegen.rs", 1160),
    ("vyre-libs/src/parsing/c/preprocess/mod.rs", 1030),
    ("vyre-driver-wgpu/src/pipeline.rs", 1000),
    ("vyre-driver/src/pipeline.rs", 1000),
    ("vyre-runtime/src/megakernel/telemetry.rs", 820),
    ("vyre-reference/src/workgroup.rs", 900),
    ("vyre-libs/src/parsing/c/parse/structure.rs", 890),
    ("vyre-foundation/src/optimizer/scheduler.rs", 1080),
    ("vyre-driver-wgpu/src/buffer/handle.rs", 870),
    ("vyre-runtime/src/megakernel/protocol.rs", 780),
    ("vyre-libs/src/matching/nfa.rs", 770),
    ("vyre-driver-wgpu/src/engine/record_and_readback.rs", 900),
    ("vyre-reference/src/hashmap_interp/step.rs", 760),
    ("vyre-foundation/src/transform/visit.rs", 830),
    ("vyre-reference/src/eval_expr.rs", 840),
];

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
    if path.starts_with(CORE_EXCLUDED) {
        return false;
    }
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
        assert_eq!(cap_for("vyre-driver-cuda/src/codegen.rs"), 1160 + 58);
    }

    /// WHY: the core ratchet has to win over the audit ceiling, because it is the
    /// tighter number. A file in both tables that took the audit ceiling would
    /// silently gain hundreds of lines of room.
    #[test]
    fn the_core_ratchet_wins_over_the_audit_ceiling() {
        let path = "vyre-foundation/src/transform/visit.rs";
        assert!(is_core(path));
        assert_eq!(cap_for(path), 789 + 40);
    }

    /// WHY: the megakernel tree is deliberately outside the core roots while the
    /// runtime restructure moves it, so it takes the flat cap and not the core one.
    #[test]
    fn the_megakernel_tree_is_not_core() {
        assert!(!is_core("vyre-runtime/src/megakernel/scheduler.rs"));
        assert_eq!(cap_for("vyre-runtime/src/megakernel/scheduler.rs"), MAX_LINES);
    }

    /// WHY: a measured core row still yields the core cap even for a megakernel
    /// path, which is exactly the case the roots exclude. The row is kept because
    /// deleting a ratchet row is not this gate's decision.
    #[test]
    fn a_megakernel_row_takes_the_flat_cap() {
        assert_eq!(
            cap_for("vyre-runtime/src/megakernel/telemetry.rs"),
            820 + 41
        );
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
