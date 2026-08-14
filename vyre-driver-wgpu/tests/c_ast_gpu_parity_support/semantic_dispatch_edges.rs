//! The switch-dispatch semantic contract, asserted against GPU buffers only.
//!
//! Selector and case-value endpoints are derived from the typed VAST tree, never
//! from the CPU semantic oracle, so these assertions also hold for a no-host
//! parser completion. Every suite that lowers a `switch` asserts the same node
//! roles and the same four dispatch edges, so the sequence has one owner here
//! and each suite supplies only its own fixture and its own extra constructs.

use super::{assert_semantic_edge, assert_semantic_node, row_indices, vast_word, SENTINEL};
use vyre_libs::parsing::c::lower::{
    C_AST_PG_CATEGORY_CONTROL, C_AST_PG_EDGE_CASE_VALUE, C_AST_PG_EDGE_SWITCH_CASE,
    C_AST_PG_EDGE_SWITCH_DEFAULT, C_AST_PG_EDGE_SWITCH_SELECTOR, C_AST_PG_ROLE_CASE,
    C_AST_PG_ROLE_DEFAULT, C_AST_PG_ROLE_SWITCH,
};
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_CASE_STMT, C_AST_KIND_DEFAULT_STMT, C_AST_KIND_SWITCH_STMT,
};

/// VAST field holding a row's next sibling.
const NEXT_SIBLING_FIELD: usize = 3;
/// VAST field holding a row's first child.
const FIRST_CHILD_FIELD: usize = 2;
/// Edge slot carrying a node's own outgoing dispatch edge.
const DISPATCH_SLOT: usize = 3;
/// Edge slot carrying the switch-to-case edge recorded on the case row.
const SWITCH_CASE_SLOT: usize = 4;

/// Rows a switch-dispatch assertion resolved, so a caller can extend the check.
pub(crate) struct SwitchRows {
    pub(crate) switch: usize,
    pub(crate) case: usize,
    pub(crate) default: usize,
}

/// The first row classified as `kind`.
///
/// # Panics
///
/// Panics when the fixture classifies no such row, which means the fixture no
/// longer exercises the construct the caller named.
pub(crate) fn first_row(typed: &[u8], kind: u32, construct: &str) -> usize {
    row_indices(typed, kind)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("fixture must classify {construct}"))
}

/// Assert the semantic roles and the four dispatch edges a `switch` lowers to.
///
/// # Panics
///
/// Panics when the fixture carries no switch, case or default row, or when the
/// VAST tree gives the switch no condition group or the case no value
/// expression, because then the edge endpoints under test do not exist.
pub(crate) fn assert_switch_dispatch_edges(typed: &[u8], nodes: &[u8], edges: &[u8]) -> SwitchRows {
    let switch = first_row(typed, C_AST_KIND_SWITCH_STMT, "a switch statement");
    let case = first_row(typed, C_AST_KIND_CASE_STMT, "a case statement");
    let default = first_row(typed, C_AST_KIND_DEFAULT_STMT, "a default statement");

    assert_semantic_node(
        nodes,
        switch,
        C_AST_KIND_SWITCH_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_SWITCH,
    );
    assert_semantic_node(
        nodes,
        case,
        C_AST_KIND_CASE_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_CASE,
    );
    assert_semantic_node(
        nodes,
        default,
        C_AST_KIND_DEFAULT_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_DEFAULT,
    );

    let condition_group = vast_word(typed, switch, NEXT_SIBLING_FIELD);
    assert_ne!(
        condition_group, SENTINEL,
        "switch must have a condition-group sibling"
    );
    let selector = vast_word(typed, condition_group as usize, FIRST_CHILD_FIELD);
    assert_ne!(
        selector, SENTINEL,
        "switch condition group must have a first-child selector"
    );
    let case_value = vast_word(typed, case, NEXT_SIBLING_FIELD);
    assert_ne!(
        case_value, SENTINEL,
        "case must have a value-expression sibling"
    );

    assert_semantic_edge(
        edges,
        switch,
        DISPATCH_SLOT,
        C_AST_PG_EDGE_SWITCH_SELECTOR,
        switch as u32,
        selector,
    );
    assert_semantic_edge(
        edges,
        case,
        DISPATCH_SLOT,
        C_AST_PG_EDGE_CASE_VALUE,
        case as u32,
        case_value,
    );
    assert_semantic_edge(
        edges,
        case,
        SWITCH_CASE_SLOT,
        C_AST_PG_EDGE_SWITCH_CASE,
        switch as u32,
        case as u32,
    );
    assert_semantic_edge(
        edges,
        default,
        DISPATCH_SLOT,
        C_AST_PG_EDGE_SWITCH_DEFAULT,
        switch as u32,
        default as u32,
    );

    SwitchRows {
        switch,
        case,
        default,
    }
}
