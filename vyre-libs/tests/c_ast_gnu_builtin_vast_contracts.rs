//! VAST structural contracts for GNU builtin expressions.
//!
//! Constructs under test:
//!   * `__builtin_expect` in expression position
//!   * `__builtin_choose_expr` with constant selector
//!   * `__builtin_types_compatible_p` with two type arguments
//!   * `__builtin_constant_p` with value argument
//!   * real-header libc builtin variants that must not be rejected as unknown

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::reference_lexer::classify_raw_source;
use c_frontend::rows::row_indices as indices_with_kind;

use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_BUILTIN_BPF_CORE_INTRIN_EXPR, C_AST_KIND_BUILTIN_CHOOSE_EXPR,
    C_AST_KIND_BUILTIN_CONSTANT_P_EXPR, C_AST_KIND_BUILTIN_EXPECT_EXPR,
    C_AST_KIND_BUILTIN_LIBC_INTRIN_EXPR, C_AST_KIND_BUILTIN_TYPES_COMPATIBLE_P_EXPR,
    C_AST_KIND_BUILTIN_UNREACHABLE_STMT,
};

fn parse_source(source: &str) -> Vec<u8> {
    classify_raw_source(source.as_bytes()).typed_vast
}

#[test]
fn builtin_expect_classified_in_expression() {
    let vast = parse_source(r#"int x = __builtin_expect(a, 1);"#);
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_EXPECT_EXPR);
    assert_eq!(
        idxs.len(),
        1,
        "__builtin_expect must produce exactly one BUILTIN_EXPECT_EXPR node"
    );
}

#[test]
fn builtin_choose_expr_classified_in_expression() {
    let vast = parse_source(r#"int x = __builtin_choose_expr(1, 2, 3);"#);
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_CHOOSE_EXPR);
    assert_eq!(
        idxs.len(),
        1,
        "__builtin_choose_expr must produce exactly one BUILTIN_CHOOSE_EXPR node"
    );
}

#[test]
fn builtin_types_compatible_p_classified_in_expression() {
    let vast = parse_source(r#"int x = __builtin_types_compatible_p(int, long);"#);
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_TYPES_COMPATIBLE_P_EXPR);
    assert_eq!(
        idxs.len(),
        1,
        "__builtin_types_compatible_p must produce exactly one BUILTIN_TYPES_COMPATIBLE_P_EXPR node"
    );
}

#[test]
fn builtin_constant_p_classified_in_expression() {
    let vast = parse_source(r#"int x = __builtin_constant_p(42);"#);
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_CONSTANT_P_EXPR);
    assert_eq!(
        idxs.len(),
        1,
        "__builtin_constant_p must produce exactly one BUILTIN_CONSTANT_P_EXPR node"
    );
}

#[test]
fn builtin_unreachable_classified_in_statement() {
    let vast = parse_source(r#"void f(void) { __builtin_unreachable(); }"#);
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_UNREACHABLE_STMT);
    assert_eq!(
        idxs.len(),
        1,
        "__builtin_unreachable must produce exactly one BUILTIN_UNREACHABLE_STMT node"
    );
}

#[test]
fn real_header_libc_builtins_parse_without_unknown_builtin_rejection() {
    let vast = parse_source(
        r#"
        int f(char *p, char *q) {
            return __builtin_memchr(p, 'x', 8) != 0
                || __builtin_strnlen(q, 16) > 4
                || __builtin___memcpy_chk(p, q, 4, 8) != 0;
        }
        "#,
    );
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_LIBC_INTRIN_EXPR);
    assert!(
        idxs.len() >= 3,
        "memchr/strnlen/memcpy_chk must each classify as BUILTIN_LIBC_INTRIN_EXPR, got {}",
        idxs.len()
    );
}

#[test]
fn real_header_libm_builtins_parse_without_unknown_builtin_rejection() {
    let vast = parse_source(
        r#"
        double f(double x, double y) {
            return __builtin_sqrt(x)
                + __builtin_pow(x, y)
                + __builtin_fma(x, y, 1.0)
                + __builtin_remainder(x, y);
        }
        "#,
    );
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_LIBC_INTRIN_EXPR);
    assert_eq!(
        idxs.len(),
        4,
        "libm GNU builtins must classify as explicit libc intrinsic VAST rows"
    );
}

#[test]
fn bpf_core_builtin_preserves_distinct_vast_semantics() {
    let vast = parse_source(r#"int f(int *p) { return __builtin_preserve_access_index(*p); }"#);
    let idxs = indices_with_kind(&vast, C_AST_KIND_BUILTIN_BPF_CORE_INTRIN_EXPR);
    assert_eq!(
        idxs.len(),
        1,
        "BPF CO-RE builtin calls must not collapse into generic assumption intrinsics"
    );
}
