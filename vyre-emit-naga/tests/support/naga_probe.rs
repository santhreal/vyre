//! Structural probes over an emitted `naga` module.
//!
//! Shared by the crate's inline emitter tests and by
//! `tests/adversarial_emit_program_matrix.rs`, which includes this file
//! directly with `#[path]`: an integration test cannot reach a `#[cfg(test)]`
//! module, and these probes are not part of the crate's public surface.

// The two inclusion sites assert over different subsets of these probes.
#![allow(dead_code)]

use naga::{BinaryOperator, Block, Expression, Statement, UnaryOperator};

/// Number of statements anywhere in `block` for which `matches` holds,
/// descending into nested blocks, both if arms, and a loop's body and
/// continuing block.
///
/// Structured lowering wraps statements in `Statement::Block` at depths that
/// move whenever it changes, so a probe reading only the top level reports a
/// barrier or an atomic that is plainly emitted as absent.
pub(crate) fn count_statements(block: &Block, matches: &dyn Fn(&Statement) -> bool) -> usize {
    block
        .iter()
        .map(|statement| {
            let here = usize::from(matches(statement));
            let nested = match statement {
                Statement::Block(child) => count_statements(child, matches),
                Statement::If { accept, reject, .. } => {
                    count_statements(accept, matches) + count_statements(reject, matches)
                }
                Statement::Loop {
                    body, continuing, ..
                } => count_statements(body, matches) + count_statements(continuing, matches),
                _ => 0,
            };
            here + nested
        })
        .sum()
}

/// Whether `block` contains a `Statement::Barrier` at any depth.
pub(crate) fn block_has_barrier(block: &Block) -> bool {
    count_statements(block, &|statement| {
        matches!(statement, Statement::Barrier(_))
    }) > 0
}

/// Whether `block` contains a `Statement::Loop` at any depth.
pub(crate) fn block_has_loop(block: &Block) -> bool {
    count_statements(block, &|statement| {
        matches!(statement, Statement::Loop { .. })
    }) > 0
}

/// Whether `block` contains a `Statement::Atomic` at any depth.
pub(crate) fn block_has_atomic(block: &Block) -> bool {
    count_statements(block, &|statement| {
        matches!(statement, Statement::Atomic { .. })
    }) > 0
}

/// Number of `Statement::If` in `block` at any depth.
pub(crate) fn block_if_count(block: &Block) -> usize {
    count_statements(block, &|statement| {
        matches!(statement, Statement::If { .. })
    })
}

/// Body of the module's single compute entry point.
pub(crate) fn entry_body(module: &naga::Module) -> &Block {
    &module.entry_points[0].function.body
}

/// Whether the entry point's expression arena holds an expression for which
/// `matches` holds.
///
/// The arena is flat and unordered, so "did the emitter produce this operation
/// at all" is an arena scan rather than a walk of the statement tree.
pub(crate) fn entry_has_expression(
    module: &naga::Module,
    matches: &dyn Fn(&Expression) -> bool,
) -> bool {
    module.entry_points[0]
        .function
        .expressions
        .iter()
        .any(|(_, expression)| matches(expression))
}

/// Whether the entry point applies unary `op` to anything.
pub(crate) fn entry_has_unary(module: &naga::Module, op: UnaryOperator) -> bool {
    entry_has_expression(
        module,
        &|expression| matches!(expression, Expression::Unary { op: found, .. } if *found == op),
    )
}

/// Whether the entry point combines anything with binary `op`.
pub(crate) fn entry_has_binary(module: &naga::Module, op: BinaryOperator) -> bool {
    entry_has_expression(
        module,
        &|expression| matches!(expression, Expression::Binary { op: found, .. } if *found == op),
    )
}
