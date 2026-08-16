//! Reading operation registrations out of source text.
//!
//! The registry is built by macro at compile time, so a gate that judges it by
//! compiling the workspace stops working exactly when a structural migration
//! has the tree half-moved. This reads the same registrations out of the text
//! instead: both spellings, ids written inline or through a file-local `const`,
//! and the tier each constructor implies.

use std::collections::BTreeMap;

use crate::cfg_test::strip_cfg_test_items;
use crate::source_scan::opaque_span;

/// Tier implied by each `OperationRegistration` constructor.
///
/// `new` takes the tier as its second argument, so it is read there rather
/// than assumed. Guessing it wrong is worse than not knowing: mapping
/// `primitive` to `Library` once reported all 122 of one crate's intrinsics as
/// misplaced compositions and buried the real findings. `primitive` names the
/// owning crate, `vyre-primitives`, and builds `OperationTier::Intrinsic`.
const CONSTRUCTOR_TIERS: &[(&str, Option<&str>)] = &[
    ("::primitive(", Some("Intrinsic")),
    ("::library(", Some("Library")),
    ("::new(", None),
];

/// Extract `(op_id, tier)` for every `OperationRegistration` in one file.
///
/// Two forms exist in the tree: a struct literal with named fields, and a
/// constructor call taking the id first. Both are scanned, because a gate that
/// understands only one form silently exempts every crate that uses the other -
/// which is how 140 registrations in one crate went unjudged.
///
/// Ids appear inline or through a file-local `const`, so the scan resolves both
/// without compiling the crate. That keeps the gate usable while the tree is
/// mid-migration and a crate does not build.
///
/// Test-gated items are removed first, so a fixture registration in a
/// `#[cfg(test)]` module is not counted as a production operation.
pub fn parse_registrations(text: &str) -> Vec<(String, Option<String>)> {
    let stripped = strip_cfg_test_items(text);
    let text = stripped.as_ref();
    let consts = string_consts(text);
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("OperationRegistration") {
        let body = &rest[start..];
        let after = &body["OperationRegistration".len()..];
        let constructor = CONSTRUCTOR_TIERS.iter().find(|(call, _)| {
            after
                .trim_start()
                .starts_with(call.trim_start_matches("::"))
                || after.starts_with(call)
        });
        if let Some((call, tier)) = constructor {
            if let Some(id) = first_argument(after, call).and_then(|raw| resolve_id(raw, &consts)) {
                let tier = tier
                    .map(|tier| tier.to_string())
                    .or_else(|| nth_argument(after, call, 1).map(tier_variant));
                found.push((id, tier));
            }
        } else {
            let block = &body[..struct_literal_end(body)];
            if let Some(id) = field_value(block, "id").and_then(|raw| resolve_id(raw, &consts)) {
                found.push((id, field_value(block, "tier").map(tier_variant)));
            }
        }
        rest = after;
    }
    found
}

/// First argument of a constructor call, as written.
fn first_argument<'a>(after: &'a str, call: &str) -> Option<&'a str> {
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
fn nth_argument<'a>(after: &'a str, call: &str, index: usize) -> Option<&'a str> {
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

/// Byte offset just past the struct literal that opens in `body`.
///
/// Registration fields hold closures, so the first `}` is almost never the end
/// of the literal. Counting depth is what keeps `id:` and `tier:` inside the
/// scanned window; stopping at the first brace silently drops most
/// registrations and makes every registration rule pass on an empty set. A
/// brace inside a string, char literal, raw string or comment is text and does
/// not count.
fn struct_literal_end(body: &str) -> usize {
    let bytes = body.as_bytes();
    let mut depth = 0usize;
    let mut opened = false;
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(body, offset) {
            offset += span.get();
            continue;
        }
        match bytes[offset] {
            b'{' => {
                depth += 1;
                opened = true;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if opened && depth == 0 {
                    return offset + 1;
                }
            }
            _ => {}
        }
        offset += 1;
    }
    body.len()
}

/// Map every `const NAME: &str = "value";` in a file to its literal.
///
/// Read over the whole text rather than line by line: a long id is wrapped
/// onto the line after the `=`, and a line-bound scan resolved none of those,
/// so every registration whose id came through such a const was dropped. The
/// declared type must name `str`, which keeps `const fn` bodies and const
/// generic parameters out of the map.
fn string_consts(text: &str) -> BTreeMap<String, String> {
    const KEYWORD: &str = "const ";
    let mut consts = BTreeMap::new();
    let mut cursor = 0usize;
    while let Some(offset) = text[cursor..].find(KEYWORD) {
        let start = cursor + offset + KEYWORD.len();
        cursor = start;
        let Some(end) = text[start..].find(';') else {
            break;
        };
        let Some((declared, value)) = text[start..start + end].split_once('=') else {
            continue;
        };
        let Some((name, declared_type)) = declared.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !declared_type.contains("str")
            || name.is_empty()
            || !name
                .chars()
                .all(|character| character.is_alphanumeric() || character == '_')
        {
            continue;
        }
        if let Some(literal) = string_literal(value) {
            consts.insert(name.to_string(), literal);
        }
    }
    consts
}

fn string_literal(text: &str) -> Option<String> {
    let start = text.find('"')?;
    let rest = &text[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Read one `field: value,` from a struct literal body.
fn field_value<'a>(block: &'a str, field: &str) -> Option<&'a str> {
    for line in block.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix(field) else {
            continue;
        };
        let Some(rest) = rest.strip_prefix(':') else {
            continue;
        };
        return Some(rest.trim().trim_end_matches(','));
    }
    None
}

fn resolve_id(raw: &str, consts: &BTreeMap<String, String>) -> Option<String> {
    if let Some(literal) = string_literal(raw) {
        return Some(literal);
    }
    consts.get(raw.trim()).cloned()
}

fn tier_variant(raw: &str) -> String {
    raw.rsplit("::").next().unwrap_or(raw).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_inline_registration_id_is_parsed() {
        let parsed = parse_registrations(
            r#"
inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        tier: vyre_foundation::operation::OperationTier::Library,
        id: "vyre-foundation::hash::adler32",
    }
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-foundation::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// Registration fields hold closures. A parser that stops at the first `}`
    /// truncates the literal before `id:` and reports no registration at all,
    /// which makes every registration rule pass on an empty set.
    /// Most registrations in the tree use the constructor form. A parser that
    /// reads only the struct-literal form exempts every crate that uses it.
    #[test]
    fn a_constructor_registration_is_parsed() {
        let parsed = parse_registrations(
            r#"
const ADLER32_OP_ID: &str = "vyre-foundation::hash::adler32";

fn registration() -> OperationRegistration {
    vyre_foundation::operation::OperationRegistration::primitive(
        ADLER32_OP_ID,
        || adler32_program("input", "out", 3),
        Some(|| { vec![vec![vec![1u8]]] }),
        Some(|| vec![vec![vec![2u8]]]),
    )
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-foundation::hash::adler32".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    /// `new` carries the tier in argument two. Assuming a tier for it once
    /// reported an entire crate's intrinsics as misplaced compositions.
    #[test]
    fn a_new_registration_reads_its_tier_argument() {
        let parsed = parse_registrations(
            r#"
    OperationRegistration::new(
        "vyre-primitives::hardware::fma_f32",
        OperationTier::Intrinsic,
        Some(fma_f32_program),
        Some(|| vec![vec![vec![1u8, 2u8]]]),
        None,
    )
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::hardware::fma_f32".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    #[test]
    fn a_constructor_registration_with_an_inline_id_is_parsed() {
        let parsed = parse_registrations(
            r#"
    OperationRegistration::library("vyre-libs::nn::attention", builder)
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::nn::attention".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_registration_whose_fields_contain_braces_is_still_parsed() {
        let parsed = parse_registrations(
            r#"
inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        build: Some(|| { let program = adler32("input", "out", 3); program }),
        test_inputs: Some(|| { vec![vec![vec![1u8]]] }),
        tier: vyre_foundation::operation::OperationTier::Library,
        id: "vyre-foundation::hash::adler32",
    }
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-foundation::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_const_backed_registration_id_is_resolved() {
        let parsed = parse_registrations(
            r#"
const OP_ID: &str = "vyre-libs::atomic::compare_exchange";

inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        tier: vyre_foundation::operation::OperationTier::Intrinsic,
        id: OP_ID,
    }
}
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::atomic::compare_exchange".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    #[test]
    fn a_registration_inside_a_test_module_is_not_a_production_operation() {
        let parsed = parse_registrations(
            r#"
            #[cfg(test)]
            mod tests {
                const ECHO_ID: &str = "test::reference_echo";
                fn fixture() {
                    OperationRegistration::library(ECHO_ID);
                }
            }
            "#,
        );

        assert_eq!(parsed, Vec::new());
    }

    #[test]
    fn a_production_registration_beside_a_test_module_is_still_counted() {
        let parsed = parse_registrations(
            r#"
            fn install() {
                OperationRegistration::library("vyre-libs::hash::crc32");
            }

            #[cfg(test)]
            mod tests {
                fn fixture() {
                    OperationRegistration::library("test::call_u32");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::crc32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_feature_named_test_something_does_not_exempt_a_registration() {
        let parsed = parse_registrations(
            r#"
            #[cfg(feature = "test-utils")]
            mod utils {
                fn install() {
                    OperationRegistration::library("vyre-libs::hash::fnv1a32");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::fnv1a32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    #[test]
    fn a_compound_test_predicate_still_strips_the_item() {
        let parsed = parse_registrations(
            r#"
            #[cfg(all(test, feature = "gpu"))]
            mod tests {
                fn fixture() {
                    OperationRegistration::library("test::reference_panic");
                }
            }
            "#,
        );

        assert_eq!(parsed, Vec::new());
    }

    #[test]
    fn a_test_gated_module_declaration_strips_only_the_declaration() {
        let parsed = parse_registrations(
            r#"
            #[cfg(test)]
            mod tests;

            fn install() {
                OperationRegistration::library("vyre-libs::hash::adler32");
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// Shape of `vyre-libs/src/hash/adler32.rs`: a braced `use` list, a const
    /// id, two test-gated `use` lines, a test-gated helper, then the real
    /// struct-literal registration, then the test module. The production id
    /// must survive all of it.
    #[test]
    fn a_production_registration_survives_a_file_full_of_test_gated_items() {
        let parsed = parse_registrations(
            r#"
            use vyre_libs::hash::adler32::{adler32_program, ADLER32_OP_ID};

            #[cfg(test)]
            use crate::buffer_names::fixed_name;
            #[cfg(test)]
            use vyre_libs::hash::adler32::adler32 as adler32_cpu_reference;

            const OP_ID: &str = "vyre-libs::hash::adler32";

            #[cfg(test)]
            fn cpu_ref(input: &[u8]) -> u32 {
                adler32_cpu_reference(input)
            }

            inventory::submit! {
                vyre_foundation::operation::OperationRegistration {
                    semantic_version: 1,
                    tier: vyre_foundation::operation::OperationTier::Library,
                    id: OP_ID,
                    build: Some(|| adler32("input", "out", 3)),
                    category: None,
                }
            }

            #[cfg(test)]
            mod tests {
                use super::*;
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::adler32".to_string(),
                Some("Library".to_string())
            )]
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

    /// WHY: stripping a `#[cfg(test)]` item used to delete every byte before
    /// the last non-test `#[cfg(...)]` attribute in the file, because one
    /// cursor served both "where to search next" and "what is still uncopied".
    /// The deleted span held the `const` the id resolved through, so a real
    /// registration became no registration. This is the shape of
    /// `vyre-primitives/src/hash/adler32.rs`: a const id, a feature-gated
    /// production registration, then a test module.
    #[test]
    fn a_non_test_cfg_attribute_keeps_the_text_before_it() {
        let parsed = parse_registrations(
            r#"
            pub const ADLER32_OP_ID: &str = "vyre-primitives::hash::adler32";

            #[cfg(feature = "inventory-registry")]
            inventory::submit! {
                OperationRegistration::primitive(ADLER32_OP_ID, builder)
            }

            #[cfg(test)]
            mod tests {
                fn fixture() {
                    OperationRegistration::library("test::reference_echo");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::hash::adler32".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    /// A long id is written on the line after the `=`. A line-bound const scan
    /// resolved none of those, and the registration was dropped in silence.
    #[test]
    fn a_const_id_wrapped_onto_the_next_line_is_resolved() {
        let parsed = parse_registrations(
            r#"
            pub const I4_MATVEC_F32_SCALED_OP_ID: &str =
                "vyre-primitives::math::quantized::i4x8_matvec_f32_scaled";

            inventory::submit! {
                OperationRegistration::primitive(I4_MATVEC_F32_SCALED_OP_ID, builder)
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::math::quantized::i4x8_matvec_f32_scaled".to_string(),
                Some("Intrinsic".to_string())
            )]
        );
    }

    /// Adversarial case for the const scan: reading the whole text rather than
    /// one line at a time reaches every `const`, including one whose value only
    /// measures a string. The declared type is what says an id, so a `usize`
    /// const resolves nothing even when a well-formed id sits in its value.
    #[test]
    fn a_const_that_is_not_a_string_resolves_no_id() {
        let parsed = parse_registrations(
            r#"
            const OP_ID: usize = "vyre-libs::hash::adler32".len();

            inventory::submit! {
                OperationRegistration::primitive(OP_ID, builder)
            }
            "#,
        );

        assert_eq!(parsed, Vec::new());
    }

    /// A brace inside a string literal used to end the struct literal early,
    /// which truncated the scanned window before `id:` and dropped the
    /// registration. The literal here carries one unbalanced `}`, which is what
    /// a brace-counting scan cannot survive.
    #[test]
    fn a_brace_inside_a_string_does_not_end_the_struct_literal() {
        let parsed = parse_registrations(
            r#"
            inventory::submit! {
                vyre_foundation::operation::OperationRegistration {
                    build: Some(|| shader("fn main() }")),
                    tier: vyre_foundation::operation::OperationTier::Library,
                    id: "vyre-libs::text::format",
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::text::format".to_string(),
                Some("Library".to_string())
            )]
        );
    }

    /// The item a `#[cfg(test)]` gates ends at its matching brace, and a brace
    /// written inside a string is not that brace. Ending the module early left
    /// its tail in the scanned text, and the fixture registration in that tail
    /// was counted as a production operation.
    #[test]
    fn a_brace_inside_a_test_module_string_does_not_end_the_module_early() {
        let parsed = parse_registrations(
            r#"
            fn install() {
                OperationRegistration::library("vyre-libs::hash::crc32");
            }

            #[cfg(test)]
            mod tests {
                fn shader() -> &'static str {
                    "fn main() }"
                }

                fn fixture() {
                    OperationRegistration::library("test::reference_echo");
                }
            }
            "#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-libs::hash::crc32".to_string(),
                Some("Library".to_string())
            )]
        );
    }
}
