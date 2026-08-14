// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::cpu_enum_tag_forward_declaration::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn gpu_parity_attribute_on_function_pointer_typedef() {
    let fix = fixture_attribute_on_function_pointer_typedef();
    assert_full_pipeline_parity(&fix, "attribute_on_function_pointer_typedef");
}

// ---------------------------------------------------------------------------
// 4. __auto_type
// ---------------------------------------------------------------------------

pub(crate) fn fixture_auto_type_pointer_init() -> Fixture {
    c_tokens("__auto_type p = & x ;")
}

#[test]
pub(crate) fn cpu_auto_type_pointer_init_classifies() {
    let fix = fixture_auto_type_pointer_init();
    let typed = classify(&fix);

    assert_eq!(
        fix.tok_types[0], TOK_GNU_AUTO_TYPE,
        "__auto_type must promote"
    );
    assert_eq!(
        word_at(&typed, 0 * VAST_STRIDE_U32),
        0,
        "__auto_type specifier must stay raw syntax"
    );
    assert_eq!(
        word_at(&typed, 1 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "p must be VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 3 * VAST_STRIDE_U32),
        C_AST_KIND_UNARY_EXPR,
        "& must be unary expr"
    );
}

#[test]
pub(crate) fn gpu_parity_auto_type_pointer_init() {
    let fix = fixture_auto_type_pointer_init();
    assert_full_pipeline_parity(&fix, "auto_type_pointer_init");
}

// ---------------------------------------------------------------------------
// 5. typeof / typeof_unqual
// ---------------------------------------------------------------------------

pub(crate) fn fixture_typeof_unqual_simple() -> Fixture {
    c_tokens("__typeof_unqual__ ( int ) x ;")
}

pub(crate) fn fixture_typeof_array_declarator() -> Fixture {
    c_tokens("typeof ( int [ 4 ] ) arr [ 2 ] ;")
}
