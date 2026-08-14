// GPU/CPU parity end-to-end tests for Linux-kernel-grade C AST constructs
// that are not Linux-specific.
//
// Constructs under test:
//   * designated initializers (dot and array-subscript, nested)
//   * compound literals in assignment and call contexts
//   * deeply nested declarators (arrays of pointers to functions)
//   * asm / __attribute__ interactions on declarations
//   * labels, goto, switch/case/default, for, while, do-while
//   * typedef shadowing in nested block scopes
//   * GNU statement expressions in initializer position
//
// A missing GPU adapter is a configuration failure.

pub(crate) use crate::c_ast_gpu_parity_support::classify;
use crate::c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, build_fixture, row_indices, Fixture, FixtureToken,
};
use crate::c_frontend::spelling::c_tokens;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR, C_AST_KIND_INITIALIZER_LIST,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// struct S { int a; int b; };
/// struct S s = { .a = 1, .b = 2 };
pub(crate) fn fixture_designated_initializer_struct() -> Fixture {
    c_tokens("struct S { int a ; int b ; } ; struct S s = { . a = 1 , . b = 2 } ;")
}

/// int a[2][3] = { [0] = { [1] = 42 } };
pub(crate) fn fixture_designated_initializer_nested_array() -> Fixture {
    c_tokens("int a [ 2 ] [ 3 ] = { [ 0 ] = { [ 1 ] = 42 } } ;")
}

/// void f() {
///   int *p = (int[]){ 1, 2 };
///   g((struct S){ .x = 3 });
/// }
pub(crate) fn fixture_compound_literal() -> Fixture {
    c_tokens("void f ( ) { int * p = ( int [ ] ) { 1 , 2 } ; g ( ( struct S ) { . x = 3 } ) ; }")
}

/// int (*(*p)[3])(int);
pub(crate) fn fixture_nested_declarator() -> Fixture {
    c_tokens("int ( * ( * p ) [ 3 ] ) ( int ) ;")
}

/// __attribute__((used)) int x;
/// void f() {
///   __asm__ volatile ("nop" ::: "memory");
/// }
pub(crate) fn fixture_asm_attribute_interaction() -> Fixture {
    c_tokens(
        "__attribute__ ( ( used ) ) int x ; void f ( void ) { __asm__ volatile ( \"nop\" : : : \
         \"memory\" ) ; }",
    )
}

/// void f() {
///   for (int i = 0; i < 10; i++) {
///     while (cond) {
///       do { break; } while (0);
///       continue;
///     }
///   }
///   switch (v) {
///     case 1: goto end;
///     default: return;
///   }
///   end:;
/// }
pub(crate) fn fixture_control_flow_all() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        // for (int i = 0; i < 10; i++)
        FixtureToken::new("for", TOK_FOR),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("i", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("i", TOK_IDENTIFIER),
        FixtureToken::new("<", TOK_LT),
        FixtureToken::new("10", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("i", TOK_IDENTIFIER),
        FixtureToken::new("++", TOK_INC),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        // while (cond)
        FixtureToken::new("while", TOK_WHILE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("cond", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        // do { break; } while (0);
        FixtureToken::new("do", TOK_DO),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("break", TOK_BREAK),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("while", TOK_WHILE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        // continue;
        FixtureToken::new("continue", TOK_CONTINUE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("}", TOK_RBRACE),
        // switch (v)
        FixtureToken::new("switch", TOK_SWITCH),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("v", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        // case 1: goto end;
        FixtureToken::new("case", TOK_CASE),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("goto", TOK_GOTO),
        FixtureToken::new("end", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        // default: return;
        FixtureToken::new("default", TOK_DEFAULT),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        // end:;
        FixtureToken::new("end", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// typedef int T;
/// void f() {
///   int T = 1;
///   T++;
/// }
/// void g() {
///   T x;
/// }
pub(crate) fn fixture_typedef_shadowing() -> Fixture {
    c_tokens("typedef int T ; void f ( ) { int T = 1 ; T ++ ; } void g ( ) { T x ; }")
}

/// int x = ({ int y = 1; y + 2; });
pub(crate) fn fixture_statement_expression() -> Fixture {
    c_tokens("int x = ( { int y = 1 ; y + 2 ; } ) ;")
}

// ---------------------------------------------------------------------------
// Tests – designated initializers
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn designated_initializer_struct_parity_and_shape() {
    let fix = fixture_designated_initializer_struct();
    assert_full_pipeline_parity(&fix, "designated_initializer_struct");

    let typed = classify(&fix);
    assert!(
        !typed.is_empty(),
        "fixture must produce non-empty typed VAST"
    );

    // The outer initializer list must exist
    let lists = row_indices(&typed, C_AST_KIND_INITIALIZER_LIST);
    assert!(
        !lists.is_empty(),
        "struct initializer must contain INITIALIZER_LIST"
    );

    // Both designator assignments must exist
    let assigns = row_indices(&typed, C_AST_KIND_ASSIGN_EXPR);
    assert!(
        assigns.len() >= 2,
        "two designated assignments must exist, got {}",
        assigns.len()
    );
}

#[test]
pub(crate) fn designated_initializer_nested_array_parity_and_shape() {
    let fix = fixture_designated_initializer_nested_array();
    assert_full_pipeline_parity(&fix, "designated_initializer_nested_array");

    let typed = classify(&fix);
    assert!(
        !typed.is_empty(),
        "fixture must produce non-empty typed VAST"
    );

    // There should be nested initializer lists (outer + inner)
    let lists = row_indices(&typed, C_AST_KIND_INITIALIZER_LIST);
    assert!(
        lists.len() >= 2,
        "nested array initializer must contain at least 2 INITIALIZER_LIST rows, got {}",
        lists.len()
    );
}

// ---------------------------------------------------------------------------
// Tests – compound literals
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn compound_literal_parity_and_shape() {
    let fix = fixture_compound_literal();
    assert_full_pipeline_parity(&fix, "compound_literal");

    let typed = classify(&fix);
    assert!(
        !typed.is_empty(),
        "fixture must produce non-empty typed VAST"
    );

    // Compound literal expression rows must appear
    let compounds = row_indices(&typed, C_AST_KIND_COMPOUND_LITERAL_EXPR);
    assert!(
        compounds.len() >= 2,
        "compound literal fixture must contain at least 2 COMPOUND_LITERAL_EXPR rows, got {}",
        compounds.len()
    );

    // Initializer lists inside compound literals
    let lists = row_indices(&typed, C_AST_KIND_INITIALIZER_LIST);
    assert!(
        !lists.is_empty(),
        "compound literal must contain INITIALIZER_LIST"
    );
}
