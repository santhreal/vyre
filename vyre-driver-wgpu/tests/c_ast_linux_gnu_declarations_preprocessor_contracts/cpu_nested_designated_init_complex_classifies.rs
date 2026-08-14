// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::cpu_alignas_on_variable_stays_raw_and_classifies::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_nested_designated_init_complex_classifies() {
    let fix = fixture_nested_designated_init_complex();
    let typed = classify(&fix);

    let lists = row_indices(&typed, C_AST_KIND_INITIALIZER_LIST);
    assert_eq!(
        lists.len(),
        3,
        "outer, middle, inner initializer lists must classify; got {lists:?}"
    );

    let members = row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert!(
        members.len() >= 3,
        "dot designators must classify; got {members:?}"
    );

    let arrays = row_indices(&typed, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR);
    assert!(
        arrays.len() >= 2,
        "array designators must classify; got {arrays:?}"
    );

    let ranges = row_indices(&typed, C_AST_KIND_RANGE_DESIGNATOR_EXPR);
    assert_eq!(ranges, vec![21], "range designator ... must classify");

    let assigns = row_indices(&typed, C_AST_KIND_ASSIGN_EXPR);
    assert!(
        assigns.len() >= 3,
        "assignments in designators must classify; got {assigns:?}"
    );
}

#[test]
pub(crate) fn pg_lower_preserves_nested_designated_init_complex() {
    let fix = fixture_nested_designated_init_complex();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    for idx in row_indices(&typed, C_AST_KIND_INITIALIZER_LIST) {
        assert_pg_preserves_fixture_row(&typed, &pg, &fix, idx, C_AST_KIND_INITIALIZER_LIST);
    }
    for idx in row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR) {
        assert_pg_preserves_fixture_row(&typed, &pg, &fix, idx, C_AST_KIND_MEMBER_ACCESS_EXPR);
    }
    for idx in row_indices(&typed, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR) {
        assert_pg_preserves_fixture_row(&typed, &pg, &fix, idx, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR);
    }
    for idx in row_indices(&typed, C_AST_KIND_RANGE_DESIGNATOR_EXPR) {
        assert_pg_preserves_fixture_row(&typed, &pg, &fix, idx, C_AST_KIND_RANGE_DESIGNATOR_EXPR);
    }
}

#[test]
pub(crate) fn gpu_parity_nested_designated_init_complex() {
    let fix = fixture_nested_designated_init_complex();
    assert_full_pipeline_parity(&fix, "nested_designated_init_complex");
}

// ---------------------------------------------------------------------------
// 8. Bitfields
// ---------------------------------------------------------------------------

pub(crate) fn fixture_bitfield_mixed_with_attribute() -> Fixture {
    c_tokens(
        "struct Flags { unsigned int a : 4 ; __attribute__ ( ( packed ) ) unsigned int : 0 ; int \
         b : 8 ; } ;",
    )
}
