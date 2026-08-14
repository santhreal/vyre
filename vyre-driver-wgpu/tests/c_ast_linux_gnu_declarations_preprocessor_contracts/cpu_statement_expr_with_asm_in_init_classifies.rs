// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::gpu_parity_signal_function::*;
use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_statement_expr_with_asm_in_init_classifies() {
    let fix = fixture_statement_expr_with_asm_in_init();
    let typed = classify(&fix);

    assert_eq!(
        word_at(&typed, 1 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "x must be VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 4 * VAST_STRIDE_U32),
        node_kind::BASIC_BLOCK,
        "statement expr brace must be BASIC_BLOCK"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![8],
        "asm must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_QUALIFIER),
        vec![9],
        "volatile must be ASM_QUALIFIER"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_TEMPLATE),
        vec![11],
        "template string must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_OUTPUT_OPERAND),
        vec![14],
        "output operand paren must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASM_INPUT_OPERAND),
        vec![19],
        "input operand paren must classify"
    );
}

#[test]
pub(crate) fn gpu_parity_statement_expr_with_asm_in_init() {
    let fix = fixture_statement_expr_with_asm_in_init();
    assert_full_pipeline_parity(&fix, "statement_expr_with_asm_in_init");
}

// ---------------------------------------------------------------------------
// 13. Macro-shaped declarations
// ---------------------------------------------------------------------------

pub(crate) fn fixture_macro_shaped_declaration_define_per_cpu() -> Fixture {
    c_tokens("DEFINE_PER_CPU ( int , x ) ;")
}

pub(crate) fn fixture_macro_shaped_declaration_list_head() -> Fixture {
    c_tokens("LIST_HEAD ( name ) ;")
}

#[test]
pub(crate) fn cpu_macro_shaped_declaration_define_per_cpu_classifies() {
    let fix = fixture_macro_shaped_declaration_define_per_cpu();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, node_kind::CALL),
        vec![0],
        "DEFINE_PER_CPU must classify as CALL"
    );
}
