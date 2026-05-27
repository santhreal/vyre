//! Linux/kernel-grade C AST contracts for macro shapes, builtins, and qualifiers.
//!
//! Constructs under test:
//!   * container_of macro definition as preprocessor token stream + call-shaped usage
//!   * list_entry / list_for_each macro patterns
//!   * __builtin_expect (likely/unlikely) direct usage and macro wrapper preservation
//!   * static inline __attribute__((always_inline)) function definitions
//!   * volatile / _Atomic qualifier promotions in declarations and parameters
//!   * Linux error-label cleanup patterns (goto err; ... err: return -errno;)
//!
//! Every fixture asserts full GPU/CPU parity for build, annotate, and classify.
//! PG preservation is asserted for rows that carry semantic payload.
//! A missing GPU adapter is a configuration failure  -  tests panic loudly.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, assert_pg_preserves_row, build_fixture, kind_at, row_indices,
    run_gpu_pg_lower_with_count as run_gpu_pg_lower, Fixture, FixtureToken, VAST_STRIDE_U32,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ALIGNOF_EXPR,
    C_AST_KIND_BUILTIN_EXPECT_EXPR, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_GOTO_STMT, C_AST_KIND_IF_STMT, C_AST_KIND_LABEL_STMT, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT,
};
use vyre_primitives::predicate::node_kind;

fn node_count_from_vast(vast: &[u8]) -> u32 {
    (vast.len() / (VAST_STRIDE_U32 * 4)) as u32
}

fn lexeme_indices(fix: &Fixture, lexeme: &str) -> Vec<usize> {
    fix.tok_starts
        .iter()
        .zip(&fix.tok_lens)
        .enumerate()
        .filter_map(|(idx, (start, len))| {
            let s = *start as usize;
            let e = s.saturating_add(*len as usize);
            (fix.source.as_bytes().get(s..e) == Some(lexeme.as_bytes())).then_some(idx)
        })
        .collect()
}

fn token_indices_containing(fix: &Fixture, needle: &str) -> Vec<usize> {
    fix.tok_starts
        .iter()
        .zip(&fix.tok_lens)
        .enumerate()
        .filter_map(|(idx, (start, len))| {
            let s = *start as usize;
            let e = s.saturating_add(*len as usize);
            let slice = fix.source.as_bytes().get(s..e)?;
            slice
                .windows(needle.len())
                .any(|w| w == needle.as_bytes())
                .then_some(idx)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// #define container_of(ptr, type, member) \
///   ({ const typeof(((type *)0)->member) *__mptr = (ptr); \
///      (type *)((char *)__mptr - offsetof(type, member)); })
/// struct node n;
/// struct node *p = container_of(&n.member, struct node, member);
/// ```
fn fixture_container_of_macro_and_use() -> Fixture {
    build_fixture(&[
        FixtureToken::new(
            "#define container_of(ptr, type, member) ({ const typeof(((type *)0)->member) *__mptr = (ptr); (type *)((char *)__mptr - offsetof(type, member)); })\n",
            TOK_PREPROC,
        ),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("node", TOK_IDENTIFIER),
        FixtureToken::new("n", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("node", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("container_of", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("&", TOK_AMP),
        FixtureToken::new("n", TOK_IDENTIFIER),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("member", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("node", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("member", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// #define list_entry(ptr, type, member) container_of(ptr, type, member)
/// struct list_head head;
/// struct task_struct *t = list_entry(head.next, struct task_struct, tasks);
/// ```
fn fixture_list_entry_macro_and_use() -> Fixture {
    build_fixture(&[
        FixtureToken::new(
            "#define list_entry(ptr, type, member) container_of(ptr, type, member)\n",
            TOK_PREPROC,
        ),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("list_head", TOK_IDENTIFIER),
        FixtureToken::new("head", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("task_struct", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("t", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("list_entry", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("head", TOK_IDENTIFIER),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("next", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("task_struct", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("tasks", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int r = __builtin_expect(!!(x), 1);
/// int s = __builtin_expect(!!(y), 0);
/// ```
fn fixture_builtin_expect_direct() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("r", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("__builtin_expect", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("!", TOK_BANG),
        FixtureToken::new("!", TOK_BANG),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("__builtin_expect", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("!", TOK_BANG),
        FixtureToken::new("!", TOK_BANG),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("y", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// #define likely(x) __builtin_expect(!!(x), 1)
/// #define unlikely(x) __builtin_expect(!!(x), 0)
/// int a = likely(cond);
/// int b = unlikely(cond);
/// ```
fn fixture_likely_unlikely_macro_shapes() -> Fixture {
    build_fixture(&[
        FixtureToken::new(
            "#define likely(x) __builtin_expect(!!(x), 1)\n",
            TOK_PREPROC,
        ),
        FixtureToken::new(
            "#define unlikely(x) __builtin_expect(!!(x), 0)\n",
            TOK_PREPROC,
        ),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("likely", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("cond", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("unlikely", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("cond", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// static inline __attribute__((always_inline)) int dispatch(void) { return 0; }
/// static __attribute__((noinline)) void slow(void) { }
/// ```
fn fixture_static_inline_with_attributes() -> Fixture {
    build_fixture(&[
        FixtureToken::new("static", TOK_STATIC),
        FixtureToken::new("inline", TOK_INLINE),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("always_inline", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("dispatch", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("static", TOK_STATIC),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("noinline", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("slow", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// ```c
/// void probe(volatile unsigned long *flags, _Atomic unsigned long *state);
/// ```
fn fixture_volatile_atomic_parameters() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("probe", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("volatile", TOK_VOLATILE),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("flags", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("_Atomic", TOK_ATOMIC),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("state", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int n = _Alignof(unsigned long);
/// ```
fn fixture_alignof_initializer_expression() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("n", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("_Alignof", TOK_ALIGNOF),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int alloc(struct device *dev) {
///     if (!dev)
///         goto err_free;
///     return 0;
/// err_free:
///     kfree(dev);
///     return -1;
/// }
/// ```
fn fixture_linux_error_label_cleanup() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("alloc", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("device", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("dev", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("if", TOK_IF),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("!", TOK_BANG),
        FixtureToken::new("dev", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("goto", TOK_GOTO),
        FixtureToken::new("err_free", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("err_free", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("kfree", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("dev", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new("-", TOK_MINUS),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

// ---------------------------------------------------------------------------
// Tests  -  container_of
// ---------------------------------------------------------------------------

mod c_ast_linux_corpus_macro_builtin_and_qualifier_contracts_part1 {

    include!("__split/c_ast_linux_corpus_macro_builtin_and_qualifier_contracts_part1.rs");
}
mod c_ast_linux_corpus_macro_builtin_and_qualifier_contracts_part2 {
    include!("__split/c_ast_linux_corpus_macro_builtin_and_qualifier_contracts_part2.rs");
}
