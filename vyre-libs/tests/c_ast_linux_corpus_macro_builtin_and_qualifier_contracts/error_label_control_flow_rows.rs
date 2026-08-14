use super::*;
use crate::c_frontend::token_fixture::classify;

#[test]
fn linux_error_label_pg_preserves_control_flow_kinds() {
    let fix = fixture_linux_error_label_cleanup();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    for idx in row_indices(&typed, C_AST_KIND_IF_STMT) {
        assert_pg_preserves_row(
            &typed,
            &pg,
            &fix.tok_starts,
            &fix.tok_lens,
            idx,
            C_AST_KIND_IF_STMT,
        );
    }
    for idx in row_indices(&typed, C_AST_KIND_GOTO_STMT) {
        assert_pg_preserves_row(
            &typed,
            &pg,
            &fix.tok_starts,
            &fix.tok_lens,
            idx,
            C_AST_KIND_GOTO_STMT,
        );
    }
    for idx in row_indices(&typed, C_AST_KIND_RETURN_STMT) {
        assert_pg_preserves_row(
            &typed,
            &pg,
            &fix.tok_starts,
            &fix.tok_lens,
            idx,
            C_AST_KIND_RETURN_STMT,
        );
    }
}
