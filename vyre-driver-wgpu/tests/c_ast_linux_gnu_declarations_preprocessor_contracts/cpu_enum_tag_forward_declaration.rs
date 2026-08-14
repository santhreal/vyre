// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::cpu_typedef_shadowed_by_auto_type_variable::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_enum_tag_forward_declaration() {
    let fix = fixture_enum_tag_forward_declaration();
    let typed = classify(&fix);

    assert_eq!(
        word_at(&typed, 0 * VAST_STRIDE_U32),
        C_AST_KIND_ENUM_DECL,
        "enum keyword must classify"
    );
    assert_eq!(
        word_at(&typed, 1 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "enum tag must classify as VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 3 * VAST_STRIDE_U32),
        C_AST_KIND_ENUM_DECL,
        "second enum keyword must classify"
    );
    assert_eq!(
        word_at(&typed, 5 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "enum variable must classify as VARIABLE"
    );
}

#[test]
pub(crate) fn gpu_parity_struct_tag_forward_declaration() {
    let fix = fixture_struct_tag_forward_declaration();
    assert_full_pipeline_parity(&fix, "struct_tag_forward_declaration");
}

#[test]
pub(crate) fn gpu_parity_enum_tag_forward_declaration() {
    let fix = fixture_enum_tag_forward_declaration();
    assert_full_pipeline_parity(&fix, "enum_tag_forward_declaration");
}

// ---------------------------------------------------------------------------
// 3. GNU __attribute__
// ---------------------------------------------------------------------------

pub(crate) fn fixture_attribute_on_struct_definition() -> Fixture {
    c_tokens("struct __attribute__ ( ( packed ) ) S { int x ; } ;")
}

pub(crate) fn fixture_attribute_on_function_pointer_typedef() -> Fixture {
    c_tokens("typedef void ( * __attribute__ ( ( aligned ( 8 ) ) ) fp ) ( void ) ;")
}
