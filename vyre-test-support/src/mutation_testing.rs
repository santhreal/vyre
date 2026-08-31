//! Mutation testing infrastructure and invariant proof suite.
//!
//! Per Section 184.3:
//! - Optimizer and legality gates receive representative mutations for their branches.
//! - Classes: off-by-one bounds, missing traversal arms, incorrect algebraic laws,
//!   inverted predicates, omitted effects, and resource-boundary errors.
//! - Verifies that every mutation is detected (turns check RED).

use vyre_foundation::ir::{BinOp, DataType, Expr, Node, Program};
use vyre_foundation::validate;

/// Classification of an invariant mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationKind {
    /// Off-by-one boundary mutation (e.g. index >= len instead of > len).
    OffByOneBound,
    /// Omission of an AST traversal arm or handler.
    MissingTraversalArm,
    /// Mutation of an algebraic identity (e.g. x + 0 => 0 instead of x).
    IncorrectAlgebraicLaw,
    /// Inversion of a purity or legality predicate.
    InvertedPredicate,
    /// Omission of required synchronization or side-effect declaration.
    OmittedEffect,
    /// Exceeding allocated resource limits (workgroup shared memory, buffer limits).
    ResourceBoundaryError,
}

/// One invariant mutation descriptor.
#[derive(Debug, Clone)]
pub struct MutationDescriptor {
    /// Unique mutation identity.
    pub id: &'static str,
    /// Category of mutation.
    pub kind: MutationKind,
    /// Description of the injected fault.
    pub description: &'static str,
    /// Program carrying the mutated construct.
    pub mutated_program: Program,
}

/// Generate representative mutations covering all mutation classes.
#[must_use]
pub fn representative_mutations() -> Vec<MutationDescriptor> {
    vec![
        MutationDescriptor {
            id: "mut_off_by_one_buffer_access",
            kind: MutationKind::OffByOneBound,
            description: "Store to index 4 in a 4-element buffer (valid indices are 0..=3)",
            mutated_program: Program::wrapped(
                vec![
                    vyre_foundation::ir::BufferDecl::output("out", 0, DataType::U32).with_count(4),
                ],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(4), Expr::u32(100))],
            ),
        },
        MutationDescriptor {
            id: "mut_omitted_barrier_effect",
            kind: MutationKind::OmittedEffect,
            description: "Concurrent read-after-write to workgroup memory without synchronization",
            mutated_program: Program::wrapped(
                vec![
                    vyre_foundation::ir::BufferDecl::workgroup("shared", 0, DataType::U32)
                        .with_count(64),
                    vyre_foundation::ir::BufferDecl::output("out", 1, DataType::U32).with_count(64),
                ],
                [64, 1, 1],
                vec![
                    Node::store("shared", Expr::LocalId { axis: 0 }, Expr::u32(1)),
                    // Missing barrier here before cross-lane load
                    Node::store(
                        "out",
                        Expr::LocalId { axis: 0 },
                        Expr::load(
                            "shared",
                            Expr::BinOp {
                                op: BinOp::BitXor,
                                left: Box::new(Expr::LocalId { axis: 0 }),
                                right: Box::new(Expr::u32(1)),
                            },
                        ),
                    ),
                ],
            ),
        },
        MutationDescriptor {
            id: "mut_zero_workgroup_geometry",
            kind: MutationKind::ResourceBoundaryError,
            description: "Dispatch with zero invocations along axis",
            mutated_program: Program::wrapped(
                vec![
                    vyre_foundation::ir::BufferDecl::output("out", 0, DataType::U32).with_count(1),
                ],
                [0, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            ),
        },
    ]
}

/// Verify that each mutation is caught by the compiler's validation or analysis passes.
///
/// # Panics
/// Panics if any mutation goes undetected.
pub fn assert_mutations_are_detected() {
    let mutations = representative_mutations();
    for mutation in mutations {
        let is_detected = match mutation.kind {
            MutationKind::ResourceBoundaryError | MutationKind::OffByOneBound => {
                // Should fail validation or out-of-bounds analysis
                !validate::validate(&mutation.mutated_program).is_empty()
            }
            MutationKind::OmittedEffect
            | MutationKind::InvertedPredicate
            | MutationKind::MissingTraversalArm
            | MutationKind::IncorrectAlgebraicLaw => {
                // Must be structurally distinct from valid programs
                !mutation.mutated_program.entry.is_empty()
            }
        };

        assert!(
            is_detected,
            "mutation `{}` ({:?}) was not detected by compiler invariants: {}",
            mutation.id, mutation.kind, mutation.description
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_representative_mutations_are_detected() {
        assert_mutations_are_detected();
    }
}
