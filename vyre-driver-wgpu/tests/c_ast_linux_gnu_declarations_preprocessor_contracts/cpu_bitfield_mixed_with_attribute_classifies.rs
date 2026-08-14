// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::cpu_nested_designated_init_complex_classifies::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_bitfield_mixed_with_attribute_classifies() {
    let fix = fixture_bitfield_mixed_with_attribute();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, C_AST_KIND_BIT_FIELD_DECL),
        vec![5, 17, 21],
        "named and unnamed bitfields must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE),
        vec![9],
        "attribute must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ATTRIBUTE_PACKED),
        vec![12],
        "packed must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_STRUCT_DECL),
        vec![0],
        "struct keyword must classify"
    );
}

#[test]
pub(crate) fn pg_lower_preserves_bitfield_mixed_with_attribute() {
    let fix = fixture_bitfield_mixed_with_attribute();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    for idx in row_indices(&typed, C_AST_KIND_BIT_FIELD_DECL) {
        assert_pg_preserves_fixture_row(&typed, &pg, &fix, idx, C_AST_KIND_BIT_FIELD_DECL);
    }
    for idx in row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE) {
        assert_pg_preserves_fixture_row(&typed, &pg, &fix, idx, C_AST_KIND_GNU_ATTRIBUTE);
    }
}

#[test]
pub(crate) fn gpu_parity_bitfield_mixed_with_attribute() {
    let fix = fixture_bitfield_mixed_with_attribute();
    assert_full_pipeline_parity(&fix, "bitfield_mixed_with_attribute");
}

// ---------------------------------------------------------------------------
// 9. Flexible arrays
// ---------------------------------------------------------------------------

pub(crate) fn fixture_flexible_array_member() -> Fixture {
    c_tokens("struct S { int n ; int arr [ ] ; } ;")
}
