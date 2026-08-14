use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
fn multiple_case_same_value_both_get_switch_case_edges() {
    // void f(int x) { switch (x) { case 0: case 0: break; } }
    let fix = c_tokens("void f ( int x ) { switch ( x ) { case 0 : case 0 : break ; } }");
    let typed = classify(&fix);
    let cases = row_indices(&typed, C_AST_KIND_CASE_STMT);
    assert_eq!(
        cases,
        vec![12, 15],
        "both case statements must be classified"
    );
    let switches = row_indices(&typed, C_AST_KIND_SWITCH_STMT);
    assert_eq!(switches, vec![7], "switch must be at row 7");

    let (_nodes, edges) = semantic_lower(&typed);
    for case_idx in &cases {
        assert_eq!(
            semantic_edge_word(&edges, *case_idx, 4, 0),
            C_AST_PG_EDGE_SWITCH_CASE,
            "duplicate case must still get SWITCH_CASE edge"
        );
    }
}

#[test]
fn nested_switch_case_binds_to_innermost_switch() {
    // void f() { switch (a) { case 1: switch (b) { case 2: break; } } }
    let fix =
        c_tokens("void f ( ) { switch ( a ) { case 1 : switch ( b ) { case 2 : break ; } } }");
    let typed = classify(&fix);
    let cases = row_indices(&typed, C_AST_KIND_CASE_STMT);
    assert_eq!(cases, vec![10, 18], "outer and inner case");
    let switches = row_indices(&typed, C_AST_KIND_SWITCH_STMT);
    assert_eq!(switches, vec![5, 13], "outer and inner switch");

    let (_nodes, edges) = semantic_lower(&typed);
    // Inner case (18) must bind to inner switch (13)
    assert_eq!(
        semantic_edge_word(&edges, cases[1], 4, 1),
        switches[1] as u32,
        "inner case must bind to inner switch"
    );
}

#[test]
fn goto_forward_then_backward_across_labels() {
    // void f() { goto mid; start: goto end; mid: goto start; end: return; }
    let fix =
        c_tokens("void f ( ) { goto mid ; start : goto end ; mid : goto start ; end : return ; }");
    let typed = classify(&fix);
    let gotos = row_indices(&typed, C_AST_KIND_GOTO_STMT);
    let labels = row_indices(&typed, C_AST_KIND_LABEL_STMT);
    assert_eq!(gotos, vec![5, 10, 15], "three gotos");
    assert_eq!(labels, vec![8, 13, 18], "three labels start/mid/end");

    let (_nodes, edges) = semantic_lower(&typed);
    // goto mid (5) -> mid label (13)
    assert_eq!(
        semantic_edge_word(&edges, gotos[0], 3, 2),
        labels[1] as u32,
        "goto mid must target mid label"
    );
    // goto end (10) -> end label (18)
    assert_eq!(
        semantic_edge_word(&edges, gotos[1], 3, 2),
        labels[2] as u32,
        "goto end must target end label"
    );
    // goto start (15) -> start label (8)
    assert_eq!(
        semantic_edge_word(&edges, gotos[2], 3, 2),
        labels[0] as u32,
        "goto start must target start label"
    );
}

#[test]
fn label_colon_on_non_identifier_does_not_crash() {
    // 1: return;  -- numeric literal followed by colon (hostile but lexically valid)
    let fix = c_tokens("1 : return ;");
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);
    assert!(
        !typed.is_empty(),
        "numeric literal colon must not crash classifier"
    );
    // The colon should NOT be classified as LABEL_STMT because the preceding token is not an identifier
    let labels = row_indices(&typed, C_AST_KIND_LABEL_STMT);
    assert!(
        labels.is_empty(),
        "colon after non-identifier must not classify as LABEL_STMT"
    );
}

#[test]
fn default_before_case_in_switch_both_have_edges() {
    // void f(int x) { switch (x) { default: case 0: break; } }
    let fix = c_tokens("void f ( int x ) { switch ( x ) { default : case 0 : break ; } }");
    let typed = classify(&fix);
    let default_idx = row_indices(&typed, C_AST_KIND_DEFAULT_STMT)[0];
    let case_idx = row_indices(&typed, C_AST_KIND_CASE_STMT)[0];
    let switch_idx = row_indices(&typed, C_AST_KIND_SWITCH_STMT)[0];

    let (_nodes, edges) = semantic_lower(&typed);
    // default edge slot 3: SWITCH_DEFAULT (switch -> default)
    assert_eq!(
        semantic_edge_word(&edges, default_idx, 3, 0),
        C_AST_PG_EDGE_SWITCH_DEFAULT,
        "default must have SWITCH_DEFAULT edge"
    );
    assert_eq!(
        semantic_edge_word(&edges, default_idx, 3, 1),
        switch_idx as u32,
        "SWITCH_DEFAULT src must be switch"
    );
    // case edge slot 4: SWITCH_CASE (switch -> case)
    assert_eq!(
        semantic_edge_word(&edges, case_idx, 4, 0),
        C_AST_PG_EDGE_SWITCH_CASE,
        "case after default must still have SWITCH_CASE edge"
    );
}

#[test]
fn struct_with_field_then_malformed_semicolon_gap() {
    // struct S { int x ; ; ; int y ; };
    let fix = c_tokens("struct S { int x ; ; ; int y ; } ;");
    let typed = classify(&fix);
    let fields = row_indices(&typed, C_AST_KIND_FIELD_DECL);
    assert_eq!(
        fields,
        vec![4, 9],
        "both field identifiers must classify as FIELD_DECL"
    );
    let pg = pg_lower(&typed);
    // PG spans for field rows must be correct
    for idx in &fields {
        let pg_start = word_at(&pg, idx * 6 + 1);
        assert_eq!(
            pg_start, fix.tok_starts[*idx],
            "PG span start for field {idx} must match token start"
        );
    }
}
