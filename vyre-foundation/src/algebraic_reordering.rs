//! Whether a schedule may reorder the combines a program performs.
//!
//! A schedule that runs a phase as a work queue, across spatial partitions, or
//! as overlapped pipeline stages changes the order in which invocations combine
//! into a shared location. Reordering a combine preserves the result only when
//! the combine is associative and commutative, which is a law of the operator
//! and its element type rather than a property of the schedule.
//!
//! The laws come from [`algebraic_law_registry`](crate::algebraic_law_registry),
//! so an extension operator that registers its own laws is answered here without
//! being named. An operator with no registered law is ordered, which is the
//! answer that costs a decomposition rather than a result.
//!
//! Which statements and expressions combine at all is the recorded per-variant
//! decision in [`node_combine`](crate::visit::node_combine) and
//! [`expr_combine`](crate::visit::expr_combine), and which combine an operator
//! applies is [`vyre_spec::CombineKind`], so a new IR variant or a new operator
//! states its own answer instead of passing through here unseen.
//!
//! Where the element type of a combine is stated by the program it is read from
//! the declaration that states it. Where it is not, every element type the
//! program declares has to be exact, because a reduction in a program that
//! declares a rounding type may be reducing over one.

use vyre_spec::{AlgebraicLaw, CombineKind};

use crate::algebraic_law_registry::has_law;
use crate::extension::resolve_atomic_op;
use crate::ir::{AtomicOp, Ident, Program};
use crate::visit::{expr_combine, node_combine, walk_exprs, walk_nodes, ExprCombine, NodeCombine};

/// Whether the combines one program performs may be reordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReorderingClass {
    /// The program combines nothing across invocations, so every execution order
    /// produces the same result.
    NoCombine,
    /// Every combine the program performs is associative and commutative.
    Reassociable,
    /// At least one combine depends on the order it is applied in.
    Ordered,
}

impl ReorderingClass {
    /// Every class, for closure over the reordering answers.
    pub const ALL: &'static [Self] = &[Self::NoCombine, Self::Reassociable, Self::Ordered];

    /// Whether a schedule may reorder the combines this class describes.
    #[must_use]
    pub const fn permits_reordering(self) -> bool {
        match self {
            Self::NoCombine | Self::Reassociable => true,
            Self::Ordered => false,
        }
    }
}

/// Register both reordering laws for one op id.
macro_rules! reassociable_combine {
    ($($id:expr),+ $(,)?) => {
        $(
            inventory::submit! {
                crate::algebraic_law_registry::AlgebraicLawRegistration::new(
                    $id,
                    AlgebraicLaw::Associative,
                )
            }
            inventory::submit! {
                crate::algebraic_law_registry::AlgebraicLawRegistration::new(
                    $id,
                    AlgebraicLaw::Commutative,
                )
            }
        )+
    };
}

/// Register commutativity alone, which a rounding combine keeps.
macro_rules! commutative_combine {
    ($($id:expr),+ $(,)?) => {
        $(
            inventory::submit! {
                crate::algebraic_law_registry::AlgebraicLawRegistration::new(
                    $id,
                    AlgebraicLaw::Commutative,
                )
            }
        )+
    };
}

// Exact arithmetic reassociates. A bitwise combine is exact whatever the element
// type, so its rounding id carries the same two laws.
reassociable_combine!(
    CombineKind::Add.law_id(true),
    CombineKind::Mul.law_id(true),
    CombineKind::Min.law_id(true),
    CombineKind::Max.law_id(true),
    CombineKind::And.law_id(true),
    CombineKind::Or.law_id(true),
    CombineKind::Xor.law_id(true),
    CombineKind::And.law_id(false),
    CombineKind::Or.law_id(false),
    CombineKind::Xor.law_id(false),
);

// A rounding combine keeps commutativity and loses associativity: two orders of
// the same addends round differently, so a reordering schedule over one does not
// compute the program that was submitted.
commutative_combine!(
    CombineKind::Add.law_id(false),
    CombineKind::Mul.law_id(false),
    CombineKind::Min.law_id(false),
    CombineKind::Max.law_id(false),
);

/// Register an identity element for one op id.
macro_rules! identity_combine {
    ($($id:expr => $element:expr),+ $(,)?) => {
        $(
            inventory::submit! {
                crate::algebraic_law_registry::AlgebraicLawRegistration::new(
                    $id,
                    AlgebraicLaw::Identity { element: $element },
                )
            }
        )+
    };
}

// The element a law states is one `u32` value, so an identity is registered
// only where the same number is the identity of every element type the law id
// covers. Addition and exclusive-or leave a value unchanged when combined with
// zero, and multiplication when combined with one, at u32 and i32 alike. `Min`
// and `Max` have no such element: their identity is the extreme of the element
// type, and the same bits are the largest u32 and a negative i32. A bitwise
// `And` identity is the all-ones pattern, which is not the all-ones value of a
// boolean element.
identity_combine!(
    CombineKind::Add.law_id(true) => 0,
    CombineKind::Mul.law_id(true) => 1,
    CombineKind::Xor.law_id(true) => 0,
    CombineKind::Or.law_id(true) => 0,
);

/// Whether the registry proves one op id both associative and commutative.
fn reorderable(op_id: &str) -> bool {
    has_law(op_id, |law| matches!(law, AlgebraicLaw::Associative))
        && has_law(op_id, |law| matches!(law, AlgebraicLaw::Commutative))
}

/// Whether one combine over one element type may be reordered.
fn combine_reorderable(combine: CombineKind, exact: bool) -> bool {
    reorderable(combine.law_id(exact))
}

/// Whether the element type one buffer declares is exact, falling back to every
/// type the program declares when the program declares no such buffer.
fn buffer_is_exact(program: &Program, buffer: &Ident) -> bool {
    program
        .buffers
        .iter()
        .find(|decl| &*decl.name == buffer.as_str())
        .map_or_else(
            || every_declared_type_is_exact(program),
            |decl| decl.element.arithmetic_is_exact(),
        )
}

/// Whether every element type the program declares is exact.
fn every_declared_type_is_exact(program: &Program) -> bool {
    program
        .buffers
        .iter()
        .all(|decl| decl.element.arithmetic_is_exact())
}

/// Whether one atomic read-modify-write may be reordered.
fn atomic_reorderable(program: &Program, op: &AtomicOp, buffer: &Ident) -> bool {
    if let AtomicOp::Opaque(id) = op {
        return resolve_atomic_op(*id)
            .is_some_and(|extension| reorderable(extension.display_name()));
    }
    op.combine()
        .is_some_and(|combine| combine_reorderable(combine, buffer_is_exact(program, buffer)))
}

/// Whether the combines a program performs may be reordered by a schedule.
///
/// The walk covers nested bodies, so a combine inside a loop, a branch, or a
/// tile region is answered the same as one at the top level.
#[must_use]
pub fn reordering_class(program: &Program) -> ReorderingClass {
    let declared_exact = every_declared_type_is_exact(program);
    let mut class = ReorderingClass::NoCombine;
    let mut observe = |reorderable: bool| {
        if class == ReorderingClass::Ordered {
            return;
        }
        class = if reorderable {
            ReorderingClass::Reassociable
        } else {
            ReorderingClass::Ordered
        };
    };
    walk_exprs(program, |expr| match expr_combine(expr) {
        None => {}
        Some(ExprCombine::Atomic { op, buffer }) => {
            observe(atomic_reorderable(program, op, buffer));
        }
        Some(ExprCombine::Subgroup { op }) => {
            observe(combine_reorderable(op.combine(), declared_exact));
        }
        Some(ExprCombine::Unknown) => observe(false),
    });
    walk_nodes(program, |node| match node_combine(node) {
        None => {}
        Some(NodeCombine::Collective { op, buffer }) => {
            observe(combine_reorderable(
                op.combine(),
                buffer_is_exact(program, buffer),
            ));
        }
        Some(NodeCombine::Tile { op }) => {
            observe(combine_reorderable(op.combine(), declared_exact));
        }
        Some(NodeCombine::Unknown) => observe(false),
    });
    class
}
