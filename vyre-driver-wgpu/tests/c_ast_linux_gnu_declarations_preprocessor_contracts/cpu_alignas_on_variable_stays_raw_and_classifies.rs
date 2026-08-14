// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::cpu_typeof_unqual_simple_classifies::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_alignas_on_variable_stays_raw_and_classifies() {
    let fix = fixture_alignas_on_variable();
    let typed = classify(&fix);

    assert_eq!(fix.tok_types[0], TOK_ALIGNAS, "_Alignas must promote");
    assert_eq!(
        word_at(&typed, 0 * VAST_STRIDE_U32),
        0,
        "_Alignas must stay raw syntax"
    );
    assert_eq!(
        word_at(&typed, 5 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "x must be VARIABLE"
    );
}

#[test]
pub(crate) fn cpu_aligned_attribute_on_array_classifies() {
    let fix = fixture_aligned_attribute_on_array();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE),
        vec![0],
        "__attribute__ must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ATTRIBUTE_ALIGNED),
        vec![3],
        "aligned must classify"
    );
    assert_eq!(
        word_at(&typed, 10 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "arr must be VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 11 * VAST_STRIDE_U32),
        C_AST_KIND_ARRAY_DECL,
        "arr brackets must be ARRAY_DECL"
    );
}

#[test]
pub(crate) fn gpu_parity_aligned_attribute_on_array() {
    let fix = fixture_aligned_attribute_on_array();
    assert_full_pipeline_parity(&fix, "aligned_attribute_on_array");
}

// ---------------------------------------------------------------------------
// 7. Designated initializers
// ---------------------------------------------------------------------------

pub(crate) fn fixture_nested_designated_init_complex() -> Fixture {
    c_tokens(
        "struct Outer o = { . inner = { . arr = { [ 0 ] = 1 , [ 1 ... 3 ] = 2 } , . flag = 1 } } \
         ;",
    )
}
