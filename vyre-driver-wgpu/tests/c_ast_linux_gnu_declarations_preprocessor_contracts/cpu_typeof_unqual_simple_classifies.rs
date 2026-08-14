// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::gpu_parity_attribute_on_function_pointer_typedef::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_typeof_unqual_simple_classifies() {
    let fix = fixture_typeof_unqual_simple();
    let typed = classify(&fix);

    assert_eq!(
        fix.tok_types[0], TOK_GNU_TYPEOF_UNQUAL,
        "__typeof_unqual__ must promote"
    );
    assert_eq!(
        word_at(&typed, 4 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "x must be VARIABLE"
    );
}

#[test]
pub(crate) fn cpu_typeof_array_declarator_classifies() {
    let fix = fixture_typeof_array_declarator();
    let typed = classify(&fix);

    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF, "typeof must promote");
    assert_eq!(
        word_at(&typed, 7 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "arr must be VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 8 * VAST_STRIDE_U32),
        C_AST_KIND_ARRAY_DECL,
        "arr brackets must be ARRAY_DECL"
    );
}

#[test]
pub(crate) fn gpu_parity_typeof_array_declarator() {
    let fix = fixture_typeof_array_declarator();
    assert_full_pipeline_parity(&fix, "typeof_array_declarator");
}

// ---------------------------------------------------------------------------
// 6. alignas / aligned
// ---------------------------------------------------------------------------

pub(crate) fn fixture_alignas_on_variable() -> Fixture {
    c_tokens("_Alignas ( 8 ) int x ;")
}

pub(crate) fn fixture_aligned_attribute_on_array() -> Fixture {
    c_tokens("__attribute__ ( ( aligned ( 16 ) ) ) int arr [ 4 ] ;")
}
