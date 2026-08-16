// C parser contract tests for GNU `__attribute__` on statements, labels,
// and declarations inside statement expressions  -  contexts likely to break
// VAST/PG lowering.
//
// Constructs under test:
//   - `__attribute__((fallthrough))` as a statement in switch bodies
//   - `__attribute__((unused))` on a declaration inside a statement expression
//   - `__attribute__((aligned))` on a label (GNU extension)
//   - multiple attributes on a declaration inside a compound statement
//   - PG lowering preservation and GPU/CPU parity
//
// A missing GPU adapter is a configuration failure; tests do not skip.

use super::gnu_attribute_statements::*;
use crate::c_frontend::rows::{
    assert_pg_preserves_fixture_row as assert_pg_preserves_row, row_indices,
};
use crate::c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ATTRIBUTE_ALIGNED, C_AST_KIND_ATTRIBUTE_FALLTHROUGH, C_AST_KIND_ATTRIBUTE_UNUSED,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GNU_STATEMENT_EXPR, C_AST_KIND_IF_STMT,
    C_AST_KIND_LABEL_STMT, C_AST_KIND_SWITCH_STMT,
};
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_attribute_fallthrough_in_switch_classifies() {
    let fix = fixture_attribute_fallthrough_statement();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_SWITCH_STMT),
        vec![7],
        "switch must classify as SWITCH_STMT"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE).is_empty(),
        "__attribute__ must classify as GNU_ATTRIBUTE"
    );
    // The fallthrough attribute detail should be recognized if the parser
    // supports statement-level attributes.
    let fallthrough_rows = row_indices(&typed, C_AST_KIND_ATTRIBUTE_FALLTHROUGH);
    assert!(
        !fallthrough_rows.is_empty(),
        "fallthrough inside switch must classify as ATTRIBUTE_FALLTHROUGH"
    );
}

#[test]
pub(crate) fn cpu_attribute_unused_in_statement_expr_classifies() {
    let fix = fixture_attribute_unused_in_statement_expr();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GNU_STATEMENT_EXPR),
        vec![3],
        "statement-expression introducer must classify"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE).is_empty(),
        "__attribute__ inside statement expression must classify"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_ATTRIBUTE_UNUSED).is_empty(),
        "unused attribute detail must classify"
    );
    let vars = row_indices(&typed, node_kind::VARIABLE);
    assert!(
        !vars.is_empty(),
        "tmp must classify as VARIABLE; got {vars:?}"
    );
}

#[test]
pub(crate) fn cpu_attribute_aligned_on_label_classifies() {
    let fix = fixture_attribute_aligned_on_label();
    let typed = classify(&fix);
    let labels = row_indices(&typed, C_AST_KIND_LABEL_STMT);
    assert_ne!(
        labels.len(),
        0,
        "label must classify as LABEL_STMT; got {labels:?}"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE).is_empty(),
        "__attribute__ before label must classify"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_ATTRIBUTE_ALIGNED).is_empty(),
        "aligned attribute detail must classify"
    );
}

#[test]
pub(crate) fn cpu_multiple_attributes_in_compound_classifies() {
    let fix = fixture_multiple_attributes_in_compound();
    let typed = classify(&fix);
    let attrs = row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE);
    assert_eq!(
        attrs.len(),
        2,
        "both __attribute__ lists must classify as GNU_ATTRIBUTE"
    );
    let vars = row_indices(&typed, node_kind::VARIABLE);
    assert!(
        !vars.is_empty(),
        "sym must classify as VARIABLE; got {vars:?}"
    );
}

#[test]
pub(crate) fn cpu_attribute_on_if_arm_statement_classifies() {
    let fix = fixture_attribute_on_if_arm_statement();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_IF_STMT),
        vec![5],
        "if must classify as IF_STMT"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE).is_empty(),
        "__attribute__ on if-arm statement must classify"
    );
}

// ---------------------------------------------------------------------------
// PG lowering preservation contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn pg_lower_preserves_attribute_fallthrough_in_switch() {
    let fix = fixture_attribute_fallthrough_statement();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 7, C_AST_KIND_SWITCH_STMT);
    for idx in row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_GNU_ATTRIBUTE);
    }
}

#[test]
pub(crate) fn pg_lower_preserves_attribute_unused_in_statement_expr() {
    let fix = fixture_attribute_unused_in_statement_expr();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 3, C_AST_KIND_GNU_STATEMENT_EXPR);
    for idx in row_indices(&typed, C_AST_KIND_ATTRIBUTE_UNUSED) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_ATTRIBUTE_UNUSED);
    }
}
