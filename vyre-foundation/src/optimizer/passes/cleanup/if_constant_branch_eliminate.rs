//! `if_constant_branch_eliminate`  -  replace `Node::If` with a constant
//! condition by inlining the surviving arm.
//!
//! Op id: `vyre-foundation::optimizer::passes::if_constant_branch_eliminate`.
//! Soundness: `Exact`  -  when the condition is a compile-time-known boolean,
//! exactly one arm executes; the other is provably dead. Inlining the live
//! arm into the parent sequence preserves observable semantics. Cost-direction:
//! monotone-down on every tracked dimension (drops the dead arm + the
//! `Node::If` itself + the condition expression). Preserves: every analysis.
//! Invalidates: nothing.
//!
//! ## Rule
//!
//! ```text
//! Node::If { cond: LitBool(true),  then, otherwise } → Node::Block(then)
//! Node::If { cond: LitBool(false), then, otherwise } → Node::Block(otherwise)
//! ```
//!
//! The wrapping `Node::Block` preserves variable scoping; the immediately
//! following `empty_block_collapse` pass (or `canonicalize`) flattens the
//! Block into the parent sequence when there's no scoping concern. We
//! emit the Block instead of splicing because:
//!   1. Splicing requires the parent sequence's mutable handle  -  our pass
//!      walks one node at a time.
//!   2. The Block-then-collapse approach is local and composable; downstream
//!      passes can rely on the post-condition that no `Node::If` has a
//!      literal-bool condition.
//!
//! ## Why a separate pass
//!
//! `const_fold` rewrites `Expr::BinOp { op: Eq, left: LitU32(7), right: LitU32(7) }`
//! → `Expr::LitBool(true)`. The constant condition then sits inside the
//! `Node::If`. This pass's job is to react: drop the now-dead arm. Without
//! it, the IR carries the dead arm forward through lowering  -  backend
//! emits unreachable code, codegen pays for the branch.

use crate::ir::{Expr, Node, Program};
use crate::optimizer::passes::driver;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};

/// Drop the dead arm of `Node::If` with a compile-time-known boolean.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "if_constant_branch_eliminate",
    requires = ["const_fold"],
    invalidates = [],
    phase = "cleanup",
    boundary_class = "abi_preserving",
    cost_model_family = "scalar"
)]
pub struct IfConstantBranchEliminatePass;

impl IfConstantBranchEliminatePass {
    /// Skip programs without any `If` whose condition is a literal bool.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        driver::analyze_candidates(
            program,
            &[crate::ir::stats::NODE_KIND_IF],
            &mut is_constant_if,
        )
    }

    /// Walk the entry tree; replace constant-condition `Node::If` with
    /// the surviving arm wrapped in a `Node::Block`.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        driver::rewrite_entry_nodes(program, &mut surviving_arm)
    }
}

/// The surviving arm of a constant-condition `If`, wrapped in a `Block`.
///
/// The `Block` keeps the arm's bindings in the scope the branch gave them,
/// which is why the dead arm can go without a scope analysis.
fn surviving_arm(node: &Node) -> Option<Vec<Node>> {
    match node {
        Node::If {
            cond: Expr::LitBool(true),
            then,
            ..
        } => Some(vec![Node::Block(then.clone())]),
        Node::If {
            cond: Expr::LitBool(false),
            otherwise,
            ..
        } => Some(vec![Node::Block(otherwise.clone())]),
        _ => None,
    }
}

/// True iff `node` is `If { cond: LitBool(_), .. }`.
fn is_constant_if(node: &Node) -> bool {
    matches!(
        node,
        Node::If {
            cond: Expr::LitBool(_),
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

    fn buf() -> BufferDecl {
        BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
    }

    fn program_with_entry(entry: Vec<Node>) -> Program {
        Program::wrapped(vec![buf()], [1, 1, 1], entry)
    }

    fn count_ifs(node: &Node) -> usize {
        match node {
            Node::If {
                then, otherwise, ..
            } => {
                1 + then.iter().map(count_ifs).sum::<usize>()
                    + otherwise.iter().map(count_ifs).sum::<usize>()
            }
            Node::Loop { body, .. } => body.iter().map(count_ifs).sum(),
            Node::Block(body) => body.iter().map(count_ifs).sum(),
            Node::Region { body, .. } => body.iter().map(count_ifs).sum(),
            _ => 0,
        }
    }

    #[test]
    fn if_true_collapses_to_then_arm() {
        let entry = vec![Node::if_then(
            Expr::bool(true),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_ifs).sum();
        assert_eq!(total, 0, "if-true must collapse; got {total} If nodes");
    }

    #[test]
    fn if_false_collapses_to_otherwise_arm() {
        let entry = vec![Node::If {
            cond: Expr::bool(false),
            then: vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
            otherwise: vec![Node::store("buf", Expr::u32(1), Expr::u32(8))],
        }];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_ifs).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn if_with_runtime_condition_kept() {
        let entry = vec![Node::if_then(
            Expr::var("c"),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(
            !result.changed,
            "If with non-literal condition must be preserved"
        );
    }

    #[test]
    fn nested_constant_ifs_all_collapse() {
        // Outer if-true containing inner if-false. Both should collapse.
        let inner = Node::If {
            cond: Expr::bool(false),
            then: vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
            otherwise: vec![Node::store("buf", Expr::u32(1), Expr::u32(8))],
        };
        let entry = vec![Node::if_then(Expr::bool(true), vec![inner])];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(result.changed);
        let total: usize = result.program.entry().iter().map(count_ifs).sum();
        assert_eq!(
            total, 0,
            "nested constant Ifs must all collapse; got {total} remaining"
        );
    }

    #[test]
    fn analyze_skips_program_with_no_constant_if() {
        let entry = vec![Node::if_then(Expr::var("c"), vec![])];
        let program = program_with_entry(entry);
        assert_eq!(
            crate::optimizer::ProgramPass::analyze(&IfConstantBranchEliminatePass, &program),
            PassAnalysis::SKIP
        );
    }

    #[test]
    fn analyze_runs_when_constant_if_present() {
        let entry = vec![Node::if_then(Expr::bool(true), vec![])];
        let program = program_with_entry(entry);
        assert_eq!(
            crate::optimizer::ProgramPass::analyze(&IfConstantBranchEliminatePass, &program),
            PassAnalysis::RUN
        );
    }

    // ── Task 2: u32/i32 truthiness coverage ────────────────────────────

    #[test]
    fn if_u32_zero_is_not_matched() {
        // LitU32(0) is falsy in WGSL but this pass only matches LitBool.
        // Verify the pass does NOT fire (negative twin).
        let entry = vec![Node::if_then(
            Expr::u32(0),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(
            !result.changed,
            "LitU32(0) is not LitBool; pass must not fire"
        );
    }

    #[test]
    fn if_u32_one_is_not_matched() {
        // LitU32(1) is truthy in WGSL but this pass only matches LitBool.
        let entry = vec![Node::if_then(
            Expr::u32(1),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(
            !result.changed,
            "LitU32(1) is not LitBool; pass must not fire"
        );
    }

    #[test]
    fn if_i32_zero_is_not_matched() {
        // LitI32(0) is falsy; pass must not fire.
        let entry = vec![Node::if_then(
            Expr::i32(0),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(
            !result.changed,
            "LitI32(0) is not LitBool; pass must not fire"
        );
    }

    #[test]
    fn if_i32_neg1_is_not_matched() {
        // LitI32(-1) is truthy (non-zero) but pass only matches LitBool.
        let entry = vec![Node::if_then(
            Expr::i32(-1),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )];
        let program = program_with_entry(entry);
        let result = IfConstantBranchEliminatePass::transform(program);
        assert!(
            !result.changed,
            "LitI32(-1) is not LitBool; pass must not fire"
        );
    }
}
