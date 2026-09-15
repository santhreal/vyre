//! The shapes every binary-operator rewrite is written in.
//!
//! A rewrite is data: the operator flag that gates it, what it demands of the
//! operands, and the action it writes. Four shapes cover every rule the kernel
//! evaluates, so a rule is one row and the IR it becomes has one owner. Written
//! out per rule instead, the same twelve lines stood forty times across the two
//! rule files, and a rule set that restates its own shape can disagree with
//! itself one operand at a time.

use vyre_foundation::ir::{Expr, Node};

use crate::optimizer::expr_arena::expr_kind;

/// Which operand of the node a rule reads.
#[derive(Clone, Copy)]
pub(super) enum Operand {
    /// The left child, at `arena_arg1`.
    Left,
    /// The right child, at `arena_arg2`.
    Right,
}

impl Operand {
    /// The binding holding this operand's encoded literal value.
    const fn value_binding(self) -> &'static str {
        match self {
            Self::Left => "l_val",
            Self::Right => "r_val",
        }
    }
}

/// The literal an operand must carry for a rule to fire.
#[derive(Clone, Copy)]
pub(super) enum Literal {
    /// A `u32` literal of this value.
    U32(u32),
    /// A `bool` literal of this value.
    Bool(bool),
}

impl Literal {
    /// The value as the arena encodes it. A bool is 1 or 0.
    const fn encoded(self) -> u32 {
        match self {
            Self::U32(value) => value,
            Self::Bool(true) => 1,
            Self::Bool(false) => 0,
        }
    }
}

/// The binding that is true when `operand` carries a literal of this type.
const fn literal_kind_binding(operand: Operand, literal: Literal) -> &'static str {
    match (operand, literal) {
        (Operand::Left, Literal::U32(_)) => "l_is_lit_u32",
        (Operand::Right, Literal::U32(_)) => "r_is_lit_u32",
        (Operand::Left, Literal::Bool(_)) => "l_is_lit_bool",
        (Operand::Right, Literal::Bool(_)) => "r_is_lit_bool",
    }
}

/// One entry of a match body: a flag the later rules read, or a rule.
///
/// The two are one sequence because a flag has to be bound before the rule
/// that gates on it, and the reader needs to see that order.
pub(super) enum Step {
    /// Bind `name` to whether the node's operator tag is `tag`.
    OperatorFlag {
        /// Binding the rules below read.
        name: &'static str,
        /// Operator tag that makes it true.
        tag: u32,
    },
    /// Write `action` when `flag` holds and `operand` is `literal`.
    LiteralIdentity {
        /// Operator flag that gates the rule.
        flag: &'static str,
        /// Operand that must be the literal.
        operand: Operand,
        /// Literal that operand must be.
        literal: Literal,
        /// Action written when the rule fires.
        action: u32,
    },
}

/// A flag step, so a table row fits on one line.
pub(super) const fn flag(name: &'static str, tag: u32) -> Step {
    Step::OperatorFlag { name, tag }
}

/// A literal-identity step, so a table row fits on one line.
pub(super) const fn literal(
    flag: &'static str,
    operand: Operand,
    value: Literal,
    action: u32,
) -> Step {
    Step::LiteralIdentity {
        flag,
        operand,
        literal: value,
        action,
    }
}

/// The IR one step becomes.
pub(super) fn step_node(step: &Step) -> Node {
    match *step {
        Step::OperatorFlag { name, tag } => operator_flag(name, tag),
        Step::LiteralIdentity {
            flag,
            operand,
            literal,
            action,
        } => literal_identity(flag, operand, literal, action),
    }
}

/// Bind `name` to whether the node's operator tag is `tag`.
pub(super) fn operator_flag(name: &'static str, tag: u32) -> Node {
    Node::let_bind(name, Expr::eq(Expr::var("op"), Expr::u32(tag)))
}

/// Write `action` when `flag` holds and `operand` is `literal`.
fn literal_identity(flag: &'static str, operand: Operand, literal: Literal, action: u32) -> Node {
    Node::if_then(
        Expr::and(
            Expr::var(flag),
            Expr::and(
                Expr::var(literal_kind_binding(operand, literal)),
                Expr::eq(
                    Expr::var(operand.value_binding()),
                    Expr::u32(literal.encoded()),
                ),
            ),
        ),
        write_action(action),
    )
}

/// Write `action` when `operator` holds and both operands are one value.
///
/// The gate is an expression rather than a flag name because three comparison
/// operators share one rule.
pub(super) fn equal_operands(operator: Expr, action: u32) -> Node {
    Node::if_then(
        Expr::and(operator, Expr::var("operands_equal")),
        write_action(action),
    )
}

/// Write `action` when the left child is an `inner_tag` node whose operand at
/// `inner_canonical` is the same value as the right child.
pub(super) fn left_child_cancellation(
    flag: &'static str,
    inner_tag: u32,
    inner_canonical: &'static str,
    action: u32,
) -> Node {
    Node::if_then(
        Expr::and(
            Expr::var(flag),
            Expr::and(
                Expr::eq(Expr::var("l_kind_full"), Expr::u32(expr_kind::BIN_OP)),
                Expr::and(
                    Expr::eq(Expr::var("l_op"), Expr::u32(inner_tag)),
                    Expr::eq(Expr::var(inner_canonical), Expr::var("can_r")),
                ),
            ),
        ),
        write_action(action),
    )
}

/// The body every rule writes: one action for the Expr under `i`.
fn write_action(action: u32) -> Vec<Node> {
    vec![Node::store(
        "rewrite_action",
        Expr::var("i"),
        Expr::u32(action),
    )]
}
