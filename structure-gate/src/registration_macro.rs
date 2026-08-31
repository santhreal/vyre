//! Parsing macro invocations and argument lists in operation registration sources.
//!
//! Hardware intrinsic registrations and definitions are often wrapped in macros
//! whose bodies expand to `OperationRegistration` calls. This module scans macro
//! invocations and splits delimiter-nested argument lists out of source text.

use std::collections::{BTreeMap, BTreeSet};

use crate::registration_text::{
    field_value, identifier_continues_at, identifier_continues_before, resolve_id,
};
use crate::source_scan::{is_word_byte, opaque_span, skip_opaque};

/// Every `macro_rules!` definition in `text`, as `(name, body)`.
///
/// A definition is `macro_rules! NAME { .. }`: the name sits between the bang
/// and the brace, so the invocation reader cannot find it and this walks the
/// braces itself. Comments and literals are stepped over, so a brace inside
/// either does not close the body early.
pub fn macro_definitions(text: &str) -> Vec<(String, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for at in crate::source_scan::code_offsets(text) {
        if !text[at..].starts_with("macro_rules") || identifier_continues_before(text, at) {
            continue;
        }
        let mut cursor = skip_opaque(text, at + "macro_rules".len());
        if cursor >= bytes.len() || bytes[cursor] != b'!' {
            continue;
        }
        cursor = skip_opaque(text, cursor + 1);
        let name_start = cursor;
        while cursor < bytes.len() && is_word_byte(bytes[cursor]) {
            cursor += 1;
        }
        let name = &text[name_start..cursor];
        cursor = skip_opaque(text, cursor);
        if name.is_empty() || cursor >= bytes.len() || bytes[cursor] != b'{' {
            continue;
        }
        let body_start = cursor + 1;
        let mut depth = 1usize;
        cursor = body_start;
        while cursor < bytes.len() {
            if let Some(span) = opaque_span(text, cursor) {
                cursor += span.get();
                continue;
            }
            match bytes[cursor] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
        out.push((name.to_string(), text[body_start..cursor].to_string()));
    }
    out
}

/// True when `text` calls `inventory::submit!` in code.
///
/// A comment explaining the linkage rule and a string literal holding the call
/// both mention it and neither submits anything, so the search runs over code
/// offsets rather than the raw text.
#[must_use]
pub fn writes_inventory_submit(text: &str) -> bool {
    crate::source_scan::code_offsets(text).any(|at| text[at..].starts_with("inventory::submit!"))
}

/// Macros whose expansion submits an inventory registration.
///
/// A crate that invokes one of these submits registrations without writing
/// `inventory::submit!` itself, so a linkage rule that reads only the literal
/// call misses it. The set is a closure over the definitions in the tree: a
/// body that writes `inventory::submit!` submits, and so does a body that
/// invokes a macro that submits. `define_unary_u32_hardware_intrinsic` reaches
/// the registry through `submit_hardware_intrinsic`, two links out.
///
/// Deriving it from the definitions rather than listing names means a macro
/// added tomorrow is judged the day it is written. A hand-kept list held four
/// of the eighteen macros this workspace already had.
///
/// What this does not see: a macro whose name is assembled by concatenation, and
/// a submission written by a procedural macro, which has no `macro_rules!` body
/// to read.
#[must_use]
pub fn submitting_macros(definitions: &BTreeMap<String, String>) -> BTreeSet<String> {
    let mut submitting: BTreeSet<String> = definitions
        .iter()
        .filter(|(_, body)| writes_inventory_submit(body))
        .map(|(name, _)| name.clone())
        .collect();
    loop {
        let reached: BTreeSet<String> = definitions
            .iter()
            .filter(|(name, body)| {
                !submitting.contains(*name)
                    && submitting
                        .iter()
                        .any(|known| !find_macro_invocations(body, known).is_empty())
            })
            .map(|(name, _)| name.clone())
            .collect();
        if reached.is_empty() {
            return submitting;
        }
        submitting.extend(reached);
    }
}

/// Read hardware registration helpers whose operation id is a named argument.
///
/// These invocations are source-level definition sites even though the
/// `OperationRegistration` constructor is expanded from their macro bodies.
pub fn parse_named_intrinsic_macros(
    text: &str,
    consts: &BTreeMap<String, String>,
    found: &mut Vec<(String, Option<String>)>,
) {
    for macro_name in ["submit_hardware_intrinsic", "submit_intrinsic_operation"] {
        for (body, _) in find_macro_invocations(text, macro_name) {
            if let Some(id) = field_value(body, "id").and_then(|raw| resolve_id(raw, consts)) {
                found.push((id, Some("Intrinsic".to_string())));
            }
        }
    }
}

/// Read hardware definition helpers whose second positional argument is the id.
pub fn parse_positional_intrinsic_macros(
    text: &str,
    consts: &BTreeMap<String, String>,
    found: &mut Vec<(String, Option<String>)>,
) {
    for macro_name in [
        "define_unary_u32_hardware_intrinsic",
        "define_barrier_u32_hardware_intrinsic",
    ] {
        for (body, _) in find_macro_invocations(text, macro_name) {
            if let Some(id) = nth_argument_in_body(body, 1).and_then(|raw| resolve_id(raw, consts))
            {
                found.push((id, Some("Intrinsic".to_string())));
            }
        }
    }
}

/// Find all invocations of a macro in source text, returning `(body, next_offset)`.
///
/// Skips comments and string literals, properly matching delimiter pairs (`()`, `{}`, `[]`).
pub fn find_macro_invocations<'a>(text: &'a str, macro_name: &str) -> Vec<(&'a str, usize)> {
    let mut invocations = Vec::new();
    let bytes = text.as_bytes();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(text, offset) {
            offset += span.get();
            continue;
        }
        if !identifier_continues_before(text, offset)
            && bytes[offset..].starts_with(macro_name.as_bytes())
        {
            let after_name = offset + macro_name.len();
            if !identifier_continues_at(text, after_name) {
                let mut cursor = after_name;
                cursor = skip_opaque(text, cursor);
                if cursor < bytes.len() && bytes[cursor] == b'!' {
                    cursor += 1;
                    cursor = skip_opaque(text, cursor);
                    if cursor < bytes.len()
                        && (bytes[cursor] == b'(' || bytes[cursor] == b'{' || bytes[cursor] == b'[')
                    {
                        let open_delim = bytes[cursor];
                        let close_delim = match open_delim {
                            b'(' => b')',
                            b'{' => b'}',
                            _ => b']',
                        };
                        let body_start = cursor + 1;
                        let mut depth = 1usize;
                        let mut body_cursor = body_start;
                        while body_cursor < bytes.len() {
                            if let Some(span) = opaque_span(text, body_cursor) {
                                body_cursor += span.get();
                                continue;
                            }
                            let c = bytes[body_cursor];
                            if c == open_delim {
                                depth += 1;
                            } else if c == close_delim {
                                depth -= 1;
                                if depth == 0 {
                                    invocations
                                        .push((&text[body_start..body_cursor], body_cursor + 1));
                                    offset = body_cursor + 1;
                                    break;
                                }
                            }
                            body_cursor += 1;
                        }
                        if depth == 0 {
                            continue;
                        }
                    }
                }
            }
        }
        offset += 1;
    }
    invocations
}

/// Extract the `index`-th top-level comma-separated argument within a macro body.
pub fn nth_argument_in_body(body: &str, index: usize) -> Option<&str> {
    let bytes = body.as_bytes();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut argument = 0usize;
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(body, offset) {
            offset += span.get();
            continue;
        }
        match bytes[offset] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth > 0 => depth -= 1,
            b',' if depth == 0 => {
                if argument == index {
                    return Some(body[start..offset].trim());
                }
                argument += 1;
                start = offset + 1;
            }
            _ => {}
        }
        offset += 1;
    }
    if argument == index {
        let tail = body[start..].trim();
        if !tail.is_empty() {
            return Some(tail);
        }
    }
    None
}

/// First argument of a constructor call, as written.
pub fn first_argument<'a>(after: &'a str, call: &str) -> Option<&'a str> {
    nth_argument(after, call, 0)
}

/// Argument `index` of a constructor call, as written.
///
/// Splits on top-level commas only: `()`, `[]` and `{}` nest, and a comma
/// inside a string, char, raw string or comment is text rather than a
/// separator. Reading the id out of argument zero is how a registration enters
/// the gate's model, so a boundary read one argument early drops the
/// registration outright and the rules below then report a registry they never
/// saw.
///
/// `<` and `>` are ordinary characters. Counting them as delimiters was worse
/// than ignoring them: registration builders are closures, so `->`, `<` and
/// `>` appear as operators constantly, one unbalanced occurrence left the depth
/// permanently wrong, and a `->` dropped the depth far enough that the `)`
/// closing a nested `Some(` was read as the end of the whole call. The cost is
/// that a generic written with a top-level comma outside any delimiter pair -
/// a bare `Vec::<u8, Global>::new()` argument - would split; no registration in
/// the tree writes one.
pub fn nth_argument<'a>(after: &'a str, call: &str, index: usize) -> Option<&'a str> {
    let open = after.find(call)? + call.len();
    let rest = &after[open..];
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut argument = 0usize;
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(rest, offset) {
            offset += span.get();
            continue;
        }
        match bytes[offset] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth > 0 => depth -= 1,
            b')' => {
                return (argument == index).then(|| rest[start..offset].trim());
            }
            b',' if depth == 0 => {
                if argument == index {
                    return Some(rest[start..offset].trim());
                }
                argument += 1;
                start = offset + 1;
            }
            _ => {}
        }
        offset += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registration_text::parse_registrations;

    /// Registration helpers hide the constructor in their macro body. The
    /// source registry must still assign each invocation to its defining crate,
    /// while an unrelated macro carrying an `id` field remains only a mention.
    #[test]
    fn hardware_registration_macro_ids_are_parsed_without_broad_macro_matching() {
        let parsed = parse_registrations(
            r#"
const FMA_ID: &str = "vyre-primitives::hardware::fma_f32";
submit_hardware_intrinsic! {
    id: FMA_ID,
    signature: F32_TERNARY_SIGNATURE,
}
define_unary_u32_hardware_intrinsic!(
    bit_reverse_u32,
    "vyre-primitives::hardware::bit_reverse_u32",
    Expr::reverse_bits,
);
documentation_entry! {
    id: "vyre-primitives::hardware::not_a_registration",
}
"#,
        );

        assert_eq!(
            parsed,
            vec![
                (
                    "vyre-primitives::hardware::fma_f32".to_string(),
                    Some("Intrinsic".to_string())
                ),
                (
                    "vyre-primitives::hardware::bit_reverse_u32".to_string(),
                    Some("Intrinsic".to_string())
                ),
            ]
        );
    }

    #[test]
    fn hardware_registration_macro_syntax_variants_are_parsed() {
        let parsed = parse_registrations(
            r#"
const STORAGE_ID: &str = "vyre-primitives::hardware::storage_barrier";
submit_hardware_intrinsic! (
    id : "vyre-primitives::hardware::fma_f32",
    signature: F32_TERNARY_SIGNATURE,
);
define_unary_u32_hardware_intrinsic! (
    bit_reverse_u32,
    "vyre-primitives::hardware::bit_reverse_u32",
    Expr::reverse_bits,
);
define_barrier_u32_hardware_intrinsic! {
    storage_barrier,
    STORAGE_ID,
    &[10u32, 20, 30, 40],
}
/// Docs mention [`OperationRegistration`].
pub fn helper() {}
"#,
        );

        assert_eq!(
            parsed,
            vec![
                (
                    "vyre-primitives::hardware::fma_f32".to_string(),
                    Some("Intrinsic".to_string())
                ),
                (
                    "vyre-primitives::hardware::bit_reverse_u32".to_string(),
                    Some("Intrinsic".to_string())
                ),
                (
                    "vyre-primitives::hardware::storage_barrier".to_string(),
                    Some("Intrinsic".to_string())
                ),
            ]
        );
    }

    /// WHY this group: `nth_argument` decides where one constructor argument
    /// ends, and `parse_registrations` reads the operation id out of argument
    /// zero. Every shape the splitter misreads is an operation the gate never
    /// judges, and a registry the gate cannot see is a registry it reports
    /// clean. The splitter counted `<` and `>` as delimiters and read no
    /// literal or comment, so one `<`, one `->`, one quoted comma or one
    /// commented comma moved every later argument boundary.
    ///
    /// What this group does not pin: a generic written with a top-level comma
    /// outside any delimiter pair, such as a bare `Vec::<u8, Global>::new()`
    /// argument. `<` and `>` are ordinary characters on purpose, because Rust
    /// writes them as comparison, shift and return-arrow tokens far more often
    /// than as a balanced pair.
    #[test]
    fn a_top_level_comma_separates_arguments() {
        let call = "::primitive(OP_ID, builder, None)";

        assert_eq!(nth_argument(call, "::primitive(", 0), Some("OP_ID"));
        assert_eq!(nth_argument(call, "::primitive(", 1), Some("builder"));
        assert_eq!(nth_argument(call, "::primitive(", 2), Some("None"));
    }

    #[test]
    fn a_nested_call_in_the_first_argument_does_not_shift_the_count() {
        let call = r#"::new(op_id("bitset", "xor"), OperationTier::Intrinsic)"#;

        assert_eq!(
            nth_argument(call, "::new(", 0),
            Some(r#"op_id("bitset", "xor")"#)
        );
        assert_eq!(
            nth_argument(call, "::new(", 1),
            Some("OperationTier::Intrinsic")
        );
    }

    /// A single `<` used to raise the depth for the rest of the call, so every
    /// later comma read as nested and every argument after it disappeared.
    #[test]
    fn a_comparison_operator_is_not_an_opening_delimiter() {
        let call = "::library(OP_ID, |n| n < 4, None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some("|n| n < 4"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// `->` lowered the depth, so the `)` closing a nested `Some(` was read as
    /// the `)` closing the constructor: the argument came back missing its own
    /// closing paren and every later argument came back as nothing.
    #[test]
    fn a_return_arrow_is_not_a_closing_delimiter() {
        let call = "::new(OP_ID, OperationTier::Library, Some(|| -> Vec<u32> { vec![1] }), None)";

        assert_eq!(
            nth_argument(call, "::new(", 2),
            Some("Some(|| -> Vec<u32> { vec![1] })")
        );
        assert_eq!(nth_argument(call, "::new(", 3), Some("None"));
    }

    /// A generic argument carries its own comma. It is not a separator because
    /// it sits inside the parentheses of the argument it belongs to.
    #[test]
    fn a_generic_argument_comma_is_not_a_separator() {
        let call = "::new(OP_ID, OperationTier::Library, Some(pairs::<String, u32>), None)";

        assert_eq!(
            nth_argument(call, "::new(", 2),
            Some("Some(pairs::<String, u32>)")
        );
        assert_eq!(nth_argument(call, "::new(", 3), Some("None"));
    }

    #[test]
    fn a_comma_inside_a_string_literal_is_not_a_separator() {
        let call = r#"::library(OP_ID, ", ", None)"#;

        assert_eq!(nth_argument(call, "::library(", 1), Some(r#"", ""#));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_parenthesis_inside_a_string_literal_does_not_close_the_call() {
        let call = r#"::library(OP_ID, "f(", None)"#;

        assert_eq!(nth_argument(call, "::library(", 1), Some(r#""f(""#));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn an_escaped_quote_does_not_end_a_string_literal() {
        let call = r#"::library(OP_ID, "a\", b(", None)"#;

        assert_eq!(nth_argument(call, "::library(", 1), Some(r#""a\", b(""#));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_nested_closure_is_one_argument() {
        let call = "::library(OP_ID, |graph| move |node| visit(graph, node), None)";

        assert_eq!(
            nth_argument(call, "::library(", 1),
            Some("|graph| move |node| visit(graph, node)")
        );
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_raw_string_argument_is_read_whole() {
        let call = "::library(OP_ID, r#\"a, b)\"#, None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some("r#\"a, b)\"#"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// A comment sits inside the argument it interrupts, so the boundaries of
    /// the arguments after it must not move.
    #[test]
    fn a_comma_in_a_comment_is_not_a_separator() {
        let line_comment = "::library(\n    OP_ID, // one, two\n    builder,\n    None,\n)";
        let block_comment = "::library(OP_ID, /* one, two */ builder, None)";

        assert_eq!(nth_argument(line_comment, "::library(", 0), Some("OP_ID"));
        assert_eq!(nth_argument(line_comment, "::library(", 2), Some("None"));
        assert_eq!(nth_argument(block_comment, "::library(", 2), Some("None"));
    }

    #[test]
    fn a_char_literal_comma_is_not_a_separator() {
        let call = "::library(OP_ID, ',', None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some("','"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// Adversarial case for the literal scanner itself: `'static` and a loop
    /// label open no char literal, so the scanner must not swallow the text up
    /// to the next quote.
    #[test]
    fn a_lifetime_is_not_a_char_literal() {
        let call = "::library(OP_ID, |text: &'static str| text.len(), None)";

        assert_eq!(
            nth_argument(call, "::library(", 1),
            Some("|text: &'static str| text.len()")
        );
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    #[test]
    fn an_escaped_quote_char_literal_is_read_whole() {
        let call = r"::library(OP_ID, '\'', None)";

        assert_eq!(nth_argument(call, "::library(", 1), Some(r"'\''"));
        assert_eq!(nth_argument(call, "::library(", 2), Some("None"));
    }

    /// The submitting set is a closure over the tree, so a macro that reaches the
    /// registry through another macro is in it. `define_unary_u32_hardware_intrinsic`
    /// writes no `inventory::submit!` of its own; it expands
    /// `submit_hardware_intrinsic!`, which does. A list of names would have to be
    /// edited for each new helper, and the four names this workspace had listed
    /// covered eighteen macros.
    #[test]
    fn a_macro_that_submits_through_another_macro_is_in_the_set() {
        let text = r#"
macro_rules! leaf_submit {
    () => {
        inventory::submit! { Registration { id: "example" } }
    };
}

macro_rules! middle_submit {
    () => {
        leaf_submit! {}
    };
}

macro_rules! outer_submit {
    () => {
        middle_submit! {}
    };
}

macro_rules! unrelated {
    () => {
        // inventory::submit! in a comment is not a submission
        let value = "inventory::submit!";
    };
}
"#;
        let definitions: BTreeMap<String, String> = macro_definitions(text).into_iter().collect();
        assert_eq!(
            definitions.keys().cloned().collect::<Vec<_>>(),
            vec![
                "leaf_submit".to_string(),
                "middle_submit".to_string(),
                "outer_submit".to_string(),
                "unrelated".to_string()
            ],
            "Fix: the definition reader must find every `macro_rules!` in the text, or the closure \
             is computed over a partial set and a submitting macro reads as inert."
        );

        let submitting = submitting_macros(&definitions);
        assert_eq!(
            submitting,
            ["leaf_submit", "middle_submit", "outer_submit"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<String>>(),
            "Fix: the set must close over macro-to-macro submission and must not admit a macro \
             whose only mention of the registry is a comment or a literal."
        );
    }

    /// The derivation is run against the tree the gate judges, not only a
    /// fixture: an empty set would make every linkage answer negative and the
    /// rule vacuous.
    #[test]
    fn the_workspace_derivation_finds_the_hardware_registration_macros() {
        let root = crate::workspace_manifest::workspace_root();
        let mut definitions: BTreeMap<String, String> = BTreeMap::new();
        for member in crate::workspace_manifest::workspace_members(&root) {
            for path in crate::workspace_manifest::source_files(&root, &member) {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                definitions.extend(macro_definitions(&text));
            }
        }
        let submitting = submitting_macros(&definitions);

        for expected in [
            "submit_intrinsic_operation",
            "submit_hardware_intrinsic",
            "define_unary_u32_hardware_intrinsic",
            "define_barrier_u32_hardware_intrinsic",
        ] {
            assert!(
                submitting.contains(expected),
                "Fix: the derivation missed `{expected}`, which submits a registration in this \
                 tree, so a crate whose only submission is that macro reads as a non-submitter. \
                 Derived set: {submitting:?}"
            );
        }
    }
}
