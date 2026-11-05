//! Nested-body descent contracts for the analysis walk.

use vyre_foundation::ir::DataType;
use vyre_lower::analyses::analyze_coalesce;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, load_global};
use vyre_lower::{KernelBody, KernelOpKind, LiteralValue};

/// Every op kind that carries a nested body, with the operand layout its
/// variant documents and the child indices that layout names.
///
/// WHY: `child_body_operands` is exhaustive, so a NEW `KernelOpKind` cannot
/// be added without stating where its child indices begin. That says
/// nothing about an EXISTING kind filed under the wrong offset, which is
/// the same silent skip: every analysis stops descending and every report
/// still comes back clean. Moving `Region` into the no-child group left the
/// whole suite green before this test existed.
///
/// It does not catch a kind that gains a second body operand without a new
/// row here; the row count is the coverage.
fn child_carrying_kinds() -> Vec<(KernelOpKind, Vec<u32>, usize)> {
    vec![
        // [cond, then]
        (KernelOpKind::StructuredIfThen, vec![0, 0], 1),
        // [cond, then, otherwise]
        (KernelOpKind::StructuredIfThenElse, vec![0, 0, 1], 2),
        // [lo, hi, body]
        (
            KernelOpKind::StructuredForLoop {
                loop_var: "i".into(),
            },
            vec![0, 0, 0],
            1,
        ),
        // [body]
        (KernelOpKind::StructuredBlock, vec![0], 1),
        // [body]
        (
            KernelOpKind::Region {
                generator: "trace".into(),
            },
            vec![0],
            1,
        ),
    ]
}

/// A child body holding one global load, which is what the descent has to
/// reach for the access to be reported.
fn arm() -> KernelBody {
    body()
        .op(lit(0, 10))
        .op(load_global(0, 10, 11))
        .literal(LiteralValue::U32(0))
        .build()
}

#[test]
fn every_child_carrying_kind_is_descended_into() {
    for (kind, operands, arms) in child_carrying_kinds() {
        let desc = descriptor("k")
            .slot(global_rw(0, DataType::U32, "buf"))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .op(lit(0, 0))
                    .op(effect(kind.clone(), operands))
                    .children(std::iter::repeat_with(arm).take(arms))
                    .literal(LiteralValue::U32(0)),
            )
            .build();
        let report = analyze_coalesce(&desc);
        assert_eq!(
            report.sites.len(),
            arms,
            "Fix: {kind:?} names {arms} child body(ies) at the operand positions its variant documents, so the descent must reach every one of them; child_body_operands filed it under the wrong offset."
        );
    }
}
