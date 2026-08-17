//! Aho-Corasick multi-pattern scanner (companion to [`dfa_compile`]).
//!
//! Consumes a transition table built by [`super::dfa_compile`] and
//! scans `haystack` for any of the compiled patterns. Emits `1` at
//! `matches[i]` whenever the automaton accepts at position `i`.
//!
//! Layout assumptions (see `dfa_compile::CompiledDfa`):
//!
//! ```text
//! transitions[state * 256 + byte] = next_state
//! accept[state]                    = 0 unless state accepts
//! ```
//!
//! One invocation per haystack byte; each invocation walks only the
//! suffix window that can still affect matches ending at that byte.
//! Callers that know the longest pattern length should use
//! [`aho_corasick_bounded`] directly.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};


/// Build a Program that scans `haystack` (u32 per byte) for any
/// accepting state of a pre-built DFA. Buffers:
///
/// - `haystack`: ReadOnly, `u32` per byte.
/// - `transitions`: ReadOnly, `u32`  -  `state * 256 + byte → next`.
/// - `accept`: ReadOnly, `u32`  -  accept table indexed by state.
/// - `matches`: ReadWrite, `u32`  -  one slot per haystack byte, set
///   to `accept[state]` (pattern_id + 1) when the automaton accepts
///   at that offset.
#[must_use]
pub fn aho_corasick(
    haystack: &str,
    transitions: &str,
    accept: &str,
    matches: &str,
    haystack_len: u32,
    state_count: u32,
) -> Program {
    aho_corasick_bounded(
        haystack,
        transitions,
        accept,
        matches,
        haystack_len,
        state_count,
        haystack_len.max(1),
    )
}

/// Build a Program that scans `haystack` using a bounded suffix window.
///
/// `max_pattern_len` is the longest pattern encoded in the transition table.
/// A match ending at byte `i` can depend only on bytes
/// `i + 1 - max_pattern_len ..= i`, so this avoids replaying the whole prefix
/// for every output position while preserving exact AC semantics.
#[must_use]
pub fn aho_corasick_bounded(
    haystack: &str,
    transitions: &str,
    accept: &str,
    matches: &str,
    haystack_len: u32,
    state_count: u32,
    max_pattern_len: u32,
) -> Program {
    let body = crate::builder::TableStateMachineComposer::new(transitions).bounded_suffix_scan_body(
        haystack,
        accept,
        matches,
        max_pattern_len,
    );

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(accept, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count),
            BufferDecl::output(matches, 3, DataType::U32).with_count(haystack_len),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::matching::aho_corasick",
            body,
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::matching::aho_corasick",
        || {
            let patterns: [&[u8]; 1] = [b"abra"];
            let compiled = crate::matching::dfa_compile(&patterns);
            aho_corasick_bounded("haystack", "transitions", "accept", "matches", 11, compiled.accept.len() as u32, compiled.max_pattern_len)
        },
        Some(|| {
            let patterns: [&[u8]; 1] = [b"abra"];
            let compiled = crate::matching::dfa_compile(&patterns);
            let haystack = b"abracadabra";

            vec![vec![
                crate::fixture_bytes::u32_bytes(&haystack.iter().map(|&b| u32::from(b)).collect::<Vec<_>>()),
                crate::fixture_bytes::u32_bytes(&compiled.transitions),
                crate::fixture_bytes::u32_bytes(&compiled.accept),
            ]]
        }),
        Some(|| vec![
            vec![
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, ],
            ],
        ]),
    )
    .with_category("scan")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::ControlFlow;
    use vyre_foundation::ir::Node;
    use vyre_foundation::ir::Expr;
    use vyre_foundation::visit::try_for_each_node;

    /// Bounds of the first `Loop` in emission order.
    ///
    /// Descent is `try_for_each_node`, the crate-wide exhaustive traversal owner.
    /// The hand-rolled walk this replaces enumerated the nesting variants itself
    /// and ended in `_ => {}`, so a `Node` variant that gains a body would have
    /// been treated as a leaf: the walk would return `None` for a program whose
    /// only DFA loop sits inside that variant, and the assertions below would
    /// report a missing walk loop instead of checking its bounds.
    fn first_loop_bounds(nodes: &[Node]) -> Option<(&Expr, &Expr)> {
        match try_for_each_node(nodes, |node| match node {
            Node::Loop { from, to, .. } => ControlFlow::Break((from, to)),
            _ => ControlFlow::Continue(()),
        }) {
            ControlFlow::Break(bounds) => Some(bounds),
            ControlFlow::Continue(()) => None,
        }
    }

    #[test]
    fn bounded_aho_corasick_scans_suffix_window_instead_of_whole_prefix() {
        let program =
            aho_corasick_bounded("haystack", "transitions", "accept", "matches", 64, 8, 4);
        let (from, to) = first_loop_bounds(program.entry())
            .expect("Fix: bounded Aho-Corasick must emit one DFA-walk loop");
        assert!(
            matches!(from, Expr::Var(name) if name.as_ref() == "scan_start"),
            "bounded Aho-Corasick must start from the max-pattern suffix window"
        );
        assert!(
            matches!(to, Expr::BinOp { .. }),
            "bounded Aho-Corasick must end at i + 1"
        );
    }
}
