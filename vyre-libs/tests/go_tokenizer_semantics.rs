//! Go tokenizer and channel-matcher semantics.
//!
//! The corpus test next door proves the Go frontend agrees with tree-sitter on
//! four real files. This suite pins the individual rules that agreement rests
//! on, each against the smallest source that exercises it, so a regression
//! names the broken rule instead of reporting a count that is off by seven.
//!
//! Three defects motivated it, all found together and all invisible to a
//! whole-file count until the others were fixed:
//!
//! 1. The lexer allocated token slots with `atomicAdd`, so the stream came
//!    back in arrival order while every extractor reads `tok_types[t + 1]` as
//!    "the next token in the source".
//! 2. The lexer emitted no statement terminator, so two consecutive receive
//!    statements tokenized identically to one send.
//! 3. The lexer opened a string literal at every quote byte, including closing
//!    quotes, which doubled every grouped import.

#![cfg(feature = "go-parser")]
#![allow(deprecated)]

mod common;
use common::decode_u32_words;
use common::go::{pack_source as pack, run, tokenize, zeroed_u32_words as zeroed};

use vyre::ir::Expr;
use vyre_libs::parsing::go::lex::{TOK_ARROW, TOK_ASSIGN, TOK_IDENTIFIER, TOK_NEWLINE, TOK_STRING};
use vyre_libs::parsing::go::parse::ast_ops::{
    go_extract_channel_receives, go_extract_channel_sends,
};
use vyre_libs::parsing::go::parse::structure::{
    go_extract_packages_and_imports, GO_SPAN_RECORD_WORDS,
};

/// One dense token: its kind and the source text it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    kind: u32,
    text: String,
}

/// The dense token stream as (kind, source text) pairs.
fn tokens(source: &str) -> Vec<Token> {
    let dense = tokenize(source);
    let kinds = decode_u32_words(&dense.types);
    let starts = decode_u32_words(&dense.starts);
    let lens = decode_u32_words(&dense.lens);
    (0..dense.count)
        .map(|i| {
            let start = starts[i] as usize;
            let end = (start + lens[i] as usize).min(source.len());
            Token {
                kind: kinds[i],
                text: source[start..end].to_string(),
            }
        })
        .collect()
}

/// Dense tokens with the newline terminators removed, for readability.
fn tokens_without_newlines(source: &str) -> Vec<Token> {
    tokens(source)
        .into_iter()
        .filter(|token| token.kind != TOK_NEWLINE)
        .collect()
}

/// The raw dense arrays, for feeding an extractor program directly.
fn dense_arrays(source: &str) -> (Vec<u8>, Vec<u8>, Vec<u8>, usize) {
    let dense = tokenize(source);
    (dense.types, dense.starts, dense.lens, dense.count)
}

fn send_count(source: &str) -> u32 {
    let (kinds, starts, lens, count) = dense_arrays(source);
    let program = go_extract_channel_sends(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(count as u32),
        "out_ops",
        "out_counts",
    );
    let out = run(
        &program,
        vec![
            kinds,
            starts,
            lens,
            pack(source),
            zeroed(count.saturating_mul(GO_SPAN_RECORD_WORDS as usize).max(1)),
            zeroed(1),
        ],
    );
    decode_u32_words(&out[1])[0] / GO_SPAN_RECORD_WORDS
}

fn receive_count(source: &str) -> u32 {
    let (kinds, starts, lens, count) = dense_arrays(source);
    let program = go_extract_channel_receives(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(count as u32),
        "out_ops",
        "out_counts",
    );
    let out = run(
        &program,
        vec![
            kinds,
            starts,
            lens,
            pack(source),
            zeroed(count.saturating_mul(GO_SPAN_RECORD_WORDS as usize).max(1)),
            zeroed(1),
        ],
    );
    decode_u32_words(&out[1])[0] / GO_SPAN_RECORD_WORDS
}

fn import_count(source: &str) -> u32 {
    let (kinds, starts, lens, count) = dense_arrays(source);
    let program = go_extract_packages_and_imports(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(count as u32),
        "out_packages",
        "out_package_counts",
        "out_imports",
        "out_import_counts",
    );
    let out = run(
        &program,
        vec![
            kinds,
            starts,
            lens,
            pack(source),
            zeroed(count.saturating_mul(GO_SPAN_RECORD_WORDS as usize).max(1)),
            zeroed(1),
            zeroed(count.saturating_mul(GO_SPAN_RECORD_WORDS as usize).max(1)),
            zeroed(1),
        ],
    );
    decode_u32_words(&out[3])[0] / GO_SPAN_RECORD_WORDS
}

// ---------------------------------------------------------------------------
// String literals
// ---------------------------------------------------------------------------

/// A closing quote does not start a second literal.
///
/// The lexer runs one lane per byte, and a lane cannot tell an opening quote
/// from a closing one by looking at its own byte. It used to assume every
/// quote opened a literal, so the lane on the closing quote scanned forward to
/// the next quote in the file and emitted a phantom literal spanning the gap.
/// Quote parity, scanned across the whole source, is what makes the decision
/// correct.
#[test]
fn one_string_literal_produces_exactly_one_string_token() {
    let strings: Vec<Token> = tokens_without_newlines("package p\nvar s = \"fmt\"\n")
        .into_iter()
        .filter(|token| token.kind == TOK_STRING)
        .collect();
    assert_eq!(
        strings,
        vec![Token {
            kind: TOK_STRING,
            text: "\"fmt\"".to_string()
        }]
    );
}

/// Two literals produce two tokens, not four.
///
/// With every quote treated as an opening quote, an N-literal source produced
/// 2N string tokens. This is the case that made grouped imports count double.
#[test]
fn two_string_literals_produce_exactly_two_string_tokens() {
    let strings: Vec<String> = tokens_without_newlines("package p\nvar a = \"x\"\nvar b = \"y\"\n")
        .into_iter()
        .filter(|token| token.kind == TOK_STRING)
        .map(|token| token.text)
        .collect();
    assert_eq!(strings, vec!["\"x\"", "\"y\""]);
}

/// An empty literal is one token, and its closing quote is not a new one.
///
/// `""` is the tightest possible arrangement of an opening and a closing
/// quote, so it is where an off-by-one in the parity test shows first.
#[test]
fn an_empty_string_literal_is_a_single_token() {
    let strings: Vec<String> = tokens_without_newlines("package p\nvar s = \"\"\n")
        .into_iter()
        .filter(|token| token.kind == TOK_STRING)
        .map(|token| token.text)
        .collect();
    assert_eq!(strings, vec!["\"\""]);
}

/// A literal's contents are not tokens.
///
/// Every lexing rule looks at one byte in isolation, so the `x` in `"x"` was
/// emitted as an identifier alongside the string token. A phantom identifier
/// between two real tokens breaks every adjacency match the extractors perform.
#[test]
fn the_contents_of_a_string_literal_are_not_lexed_as_code() {
    let texts: Vec<String> = tokens_without_newlines("package p\nvar s = \"x\"\n")
        .into_iter()
        .map(|token| token.text)
        .collect();
    assert_eq!(texts, vec!["package", "p", "var", "s", "=", "\"x\""]);
}

/// Punctuation inside a literal is not punctuation.
///
/// The import path `"net/http"` and any dotted string would otherwise emit
/// dots and slashes that the structure extractor reads as real syntax.
#[test]
fn punctuation_inside_a_string_literal_is_not_tokenized() {
    let texts: Vec<String> = tokens_without_newlines("package p\nvar s = \"a.b(c)\"\n")
        .into_iter()
        .map(|token| token.text)
        .collect();
    assert_eq!(texts, vec!["package", "p", "var", "s", "=", "\"a.b(c)\""]);
}

/// An escaped quote does not close the literal.
///
/// `\"` is content. Treating it as a delimiter both truncates the token and,
/// worse, inverts quote parity for the whole rest of the file, so every later
/// literal is read inside out and its contents are lexed as code.
#[test]
fn an_escaped_quote_does_not_terminate_the_literal() {
    let strings: Vec<String> = tokens_without_newlines("package p\nvar s = \"say \\\"hi\\\"\"\n")
        .into_iter()
        .filter(|token| token.kind == TOK_STRING)
        .map(|token| token.text)
        .collect();
    assert_eq!(strings, vec!["\"say \\\"hi\\\"\""]);
}

/// A trailing escaped backslash still closes the literal.
///
/// In `"a\\"` the two backslashes escape each other, so the final quote IS a
/// delimiter. This is why the fix counts the run of backslashes rather than
/// testing the single preceding byte.
#[test]
fn a_doubled_backslash_does_not_escape_the_closing_quote() {
    let texts: Vec<String> = tokens_without_newlines("package p\nvar s = \"a\\\\\"\nvar t = 1\n")
        .into_iter()
        .map(|token| token.text)
        .collect();
    assert_eq!(
        texts,
        vec![
            "package",
            "p",
            "var",
            "s",
            "=",
            "\"a\\\\\"",
            "var",
            "t",
            "="
        ]
    );
}

/// Code after a literal containing an escaped quote is still lexed.
///
/// The parity-inversion consequence stated directly: if the escaped quote were
/// counted, `var` and `t` below would fall inside a phantom literal and vanish
/// from the stream.
#[test]
fn code_after_an_escaped_quote_is_still_lexed() {
    let texts: Vec<String> = tokens_without_newlines("package p\nvar s = \"\\\"\"\nvar t = 1\n")
        .into_iter()
        .map(|token| token.text)
        .collect();
    assert!(
        texts.contains(&"t".to_string()),
        "the declaration after the literal must survive: {texts:?}"
    );
}

/// An import path with a slash counts once and keeps its whole path.
///
/// The realistic form of the punctuation case: `"net/http"` is one token, and
/// the structure extractor must see exactly one import.
#[test]
fn a_slashed_import_path_is_one_import_and_one_token() {
    let source = "package p\n\nimport (\n\t\"net/http\"\n)\n";
    assert_eq!(import_count(source), 1);
    let strings: Vec<String> = tokens_without_newlines(source)
        .into_iter()
        .filter(|token| token.kind == TOK_STRING)
        .map(|token| token.text)
        .collect();
    assert_eq!(strings, vec!["\"net/http\""]);
}

// ---------------------------------------------------------------------------
// Statement terminators
// ---------------------------------------------------------------------------

/// Line breaks reach the token stream as terminators.
///
/// Go ends statements at a newline. Without a token for it, `<-a` followed by
/// `<-b` is byte-for-byte the same token sequence as `a <- b`, and the channel
/// matchers cannot tell a pair of receives from a single send.
#[test]
fn a_line_break_emits_a_terminator_token() {
    let kinds: Vec<u32> = tokens("package p\n").into_iter().map(|t| t.kind).collect();
    assert!(
        kinds.contains(&TOK_NEWLINE),
        "the newline after the package clause must be tokenized: {kinds:?}"
    );
}

/// Two receives on separate lines stay two receives.
///
/// The exact shape that miscounted: without a terminator between them the
/// second receive's channel was read as the first statement's send target.
#[test]
fn consecutive_receive_statements_are_two_receives_and_no_send() {
    let source = "package p\nfunc f() {\n<-a\n<-b\n}\n";
    assert_eq!(receive_count(source), 2, "both receives must be counted");
    assert_eq!(send_count(source), 0, "neither line is a send");
}

/// A send followed by a receive is one of each.
///
/// Numeric literals emit no token, so `out <- 1` then `<-in` reduces to
/// `IDENT ARROW ARROW IDENT`; the terminator is what keeps the two statements
/// apart.
#[test]
fn a_send_then_a_receive_is_one_of_each() {
    let source = "package p\nfunc f() {\nout <- 1\n<-in\n}\n";
    assert_eq!(send_count(source), 1);
    assert_eq!(receive_count(source), 1);
}

// ---------------------------------------------------------------------------
// Channel types are not channel operations
// ---------------------------------------------------------------------------

/// A receive-only channel parameter is not a send.
///
/// `in <-chan int` is `IDENTIFIER ARROW ...`, the same prefix as the send
/// `in <- value`. The `chan` keyword after the arrow is what distinguishes
/// them.
#[test]
fn a_receive_only_channel_parameter_is_not_a_send() {
    let source = "package p\nfunc f(in <-chan int) {\n}\n";
    assert_eq!(send_count(source), 0);
    assert_eq!(receive_count(source), 0);
}

/// A send-only channel parameter is neither a send nor a receive.
///
/// `out chan<- int` is `IDENTIFIER IDENTIFIER ARROW IDENTIFIER`: the arrow is
/// preceded by `chan` and followed by a type name, so a naive receive matcher
/// claims it.
#[test]
fn a_send_only_channel_parameter_is_not_an_operation() {
    let source = "package p\nfunc f(out chan<- int) {\n}\n";
    assert_eq!(send_count(source), 0);
    assert_eq!(receive_count(source), 0);
}

/// Both directional channel types in one signature stay silent.
///
/// The interface method that first exposed this: two channel types, no
/// operations, and four tokens that look like operations.
#[test]
fn a_signature_with_both_channel_directions_reports_no_operations() {
    let source = "package p\ntype S interface {\nExecute(<-chan int, chan<- int)\n}\n";
    assert_eq!(send_count(source), 0);
    assert_eq!(receive_count(source), 0);
}

/// A real send on a channel-typed parameter is still counted.
///
/// The negative twin of the two tests above. Suppressing channel types must
/// not suppress the operations performed on those channels, which is the
/// failure mode a fix that simply ignored anything near `chan` would have.
#[test]
fn a_send_on_a_directional_channel_parameter_is_counted() {
    let source = "package p\nfunc f(out chan<- int) {\nout <- 1\n}\n";
    assert_eq!(send_count(source), 1);
    assert_eq!(receive_count(source), 0);
}

/// A receive from a receive-only parameter is still counted.
#[test]
fn a_receive_from_a_directional_channel_parameter_is_counted() {
    let source = "package p\nfunc f(in <-chan int) {\n<-in\n}\n";
    assert_eq!(send_count(source), 0);
    assert_eq!(receive_count(source), 1);
}

// ---------------------------------------------------------------------------
// Keywords in front of a receive
// ---------------------------------------------------------------------------

/// `return <-ch` is a receive, not a send.
///
/// `return` lexes as an identifier, so the shape is `IDENTIFIER ARROW
/// IDENTIFIER`, identical to a send. Recognising the keyword is what puts it
/// on the right side.
#[test]
fn a_returned_receive_is_a_receive() {
    let source = "package p\nfunc f() int {\nreturn <-ch\n}\n";
    assert_eq!(receive_count(source), 1);
    assert_eq!(send_count(source), 0);
}

/// A short variable declaration from a channel is a receive.
#[test]
fn a_short_declaration_from_a_channel_is_a_receive() {
    let source = "package p\nfunc f() {\nv := <-ch\n}\n";
    assert_eq!(receive_count(source), 1);
    assert_eq!(send_count(source), 0);
}

/// Sending a received value is one send and one receive.
///
/// `out <- <-in` puts both operations in one statement, with the send's arrow
/// immediately followed by the receive's.
#[test]
fn forwarding_a_received_value_is_one_send_and_one_receive() {
    let source = "package p\nfunc f() {\nout <- <-in\n}\n";
    assert_eq!(send_count(source), 1);
    assert_eq!(receive_count(source), 1);
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

/// A single ungrouped import counts once.
#[test]
fn an_ungrouped_import_counts_once() {
    assert_eq!(import_count("package p\n\nimport \"time\"\n"), 1);
}

/// A grouped import with one spec counts once, not twice.
///
/// This is the case the phantom string literal doubled.
#[test]
fn a_grouped_import_with_one_spec_counts_once() {
    assert_eq!(import_count("package p\n\nimport (\n\t\"fmt\"\n)\n"), 1);
}

/// A grouped import counts each spec exactly once.
#[test]
fn a_grouped_import_counts_each_spec_once() {
    assert_eq!(
        import_count("package p\n\nimport (\n\t\"context\"\n\t\"fmt\"\n\t\"time\"\n)\n"),
        3
    );
}

/// String literals after the import block are not counted as imports.
///
/// The grouped scan stops at the closing parenthesis. If that termination
/// broke, every quoted string in the file would be counted as an import, which
/// is a much larger over-count than the doubling and worth separating.
#[test]
fn string_literals_elsewhere_in_the_file_are_not_imports() {
    let source =
        "package p\n\nimport (\n\t\"fmt\"\n)\n\nfunc f() {\n\tlog(\"a\")\n\tlog(\"b\")\n}\n";
    assert_eq!(import_count(source), 1);
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

/// The dense stream is in source order, kind by kind.
///
/// A direct statement of the property that the atomic compaction violated,
/// checked against a source whose expected token sequence is written out in
/// full rather than summarized as a count.
#[test]
fn the_token_stream_matches_the_source_order_exactly() {
    let actual: Vec<(u32, String)> = tokens_without_newlines("package p\nvar s = \"x\"\n")
        .into_iter()
        .map(|token| (token.kind, token.text))
        .collect();
    assert_eq!(
        actual,
        vec![
            (TOK_IDENTIFIER, "package".to_string()),
            (TOK_IDENTIFIER, "p".to_string()),
            (TOK_IDENTIFIER, "var".to_string()),
            (TOK_IDENTIFIER, "s".to_string()),
            (TOK_ASSIGN, "=".to_string()),
            (TOK_STRING, "\"x\"".to_string()),
        ]
    );
}

/// An arrow lexes as one token, not two comparison bytes.
///
/// Supporting check for every channel test above: if `<-` split into `<` and
/// `-` the matchers would see no arrows at all.
#[test]
fn a_channel_arrow_is_a_single_token() {
    let arrows: Vec<String> = tokens_without_newlines("package p\nfunc f() {\nout <- 1\n}\n")
        .into_iter()
        .filter(|token| token.kind == TOK_ARROW)
        .map(|token| token.text)
        .collect();
    assert_eq!(arrows, vec!["<-"]);
}
