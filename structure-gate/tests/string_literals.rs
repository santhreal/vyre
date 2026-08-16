//! Reading the string literals out of Rust source text.
//!
//! WHY: the operation schema reads which file names an operation id, and its
//! reader tracked only the double quote. A lexer holds `b'"'`, which opened a
//! string that never closed, so every literal after it was read as the gap
//! between two later quotes and the registration id in
//! `vyre-libs/src/parsing/python/lex.rs` was never seen. The operation the file
//! defines was then reported as having no definition site.
//!
//! What these do not catch: a literal built by concatenation or by a macro. The
//! reader answers what the text holds, not what a build produces.

use structure_gate::source_scan::string_literals;

/// A char literal holding a quote does not open a string.
#[test]
fn a_quote_inside_a_char_literal_does_not_swallow_the_rest_of_the_file() {
    let source =
        "table[usize::from(b'\"')] = C_DQUOTE;\nconst ID: &str = \"vyre-libs::parsing::lex\";\n";

    assert_eq!(string_literals(source), vec!["vyre-libs::parsing::lex"]);
}

/// A quote inside a comment is not a delimiter.
#[test]
fn a_quote_in_a_comment_is_not_a_delimiter() {
    let source =
        "// the id is \"quoted here\n/* and \"here */\nconst ID: &str = \"libs::math::matmul\";\n";

    assert_eq!(string_literals(source), vec!["libs::math::matmul"]);
}

/// Raw, byte and C string forms are read through their delimiters.
#[test]
fn every_literal_form_is_read_through_its_delimiters() {
    let source = "let a = r\"raw\";\nlet b = r#\"hash \"inside\" raw\"#;\nlet c = b\"bytes\";\nlet d = c\"cstr\";\n";

    assert_eq!(
        string_literals(source),
        vec!["raw", "hash \"inside\" raw", "bytes", "cstr"]
    );
}

/// An escaped quote stays inside its literal.
#[test]
fn an_escaped_quote_does_not_close_a_literal() {
    let source = "let a = \"x\\\"y\";\nlet b = \"after\";\n";

    assert_eq!(string_literals(source), vec!["x\\\"y", "after"]);
}

/// A lifetime is not a char literal, and does not hide the literal after it.
#[test]
fn a_lifetime_does_not_hide_the_literal_after_it() {
    let source = "fn f<'a>(x: &'a str) -> &'a str { \"kept\" }\n";

    assert_eq!(string_literals(source), vec!["kept"]);
}

/// An unterminated literal reports nothing rather than the rest of the file.
#[test]
fn an_unterminated_literal_reports_nothing() {
    let source = "const ID: &str = \"never closed\n";

    assert!(string_literals(source).is_empty());
}
