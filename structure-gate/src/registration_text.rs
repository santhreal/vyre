//! Reading operation registrations out of source text.
//!
//! The registry is built by macro at compile time, so a gate that judges it by
//! compiling the workspace stops working exactly when a structural migration
//! has the tree half-moved. This reads the same registrations out of the text
//! instead: both spellings, ids written inline or through a file-local `const`,
//! and the tier each constructor implies.

use std::collections::BTreeMap;

use crate::cfg_test::strip_cfg_test_items;
use crate::registration_macro::{
    first_argument, nth_argument, parse_named_intrinsic_macros, parse_positional_intrinsic_macros,
};
use crate::source_scan::{is_word_byte, mask_comments_and_strings, opaque_span};
/// Tier implied by each `OperationRegistration` constructor.
///
/// `new` takes the tier as its second argument, so it is read there rather
/// than assumed. Guessing it wrong is worse than not knowing: mapping
/// `primitive` to `Library` once reported all 122 of one crate's intrinsics as
/// misplaced compositions and buried the real findings. `primitive` names the
/// owning crate, `vyre-primitives`, and builds `OperationTier::Intrinsic`.
const CONSTRUCTOR_TIERS: &[(&str, Option<&str>)] = &[
    ("::primitive(", Some("Intrinsic")),
    ("::intrinsic(", Some("Intrinsic")),
    ("::library(", Some("Library")),
    ("::new(", None),
];

/// Treat every non-ASCII scalar conservatively as an identifier continuation.
///
/// Rust identifiers admit Unicode. Looking only at the adjacent byte mistakes
/// the ASCII suffix of `λOperationRegistration` for the canonical type name.

fn is_comment_span(text: &str, at: usize) -> Option<usize> {
    let rest = text.get(at..)?;
    if !rest.starts_with("//") && !rest.starts_with("/*") {
        return None;
    }
    opaque_span(text, at).map(std::num::NonZeroUsize::get)
}
pub(crate) fn identifier_continues_before(text: &str, at: usize) -> bool {
    text.get(..at)
        .and_then(|prefix| prefix.chars().next_back())
        .is_some_and(identifier_continuation)
}

pub(crate) fn identifier_continues_at(text: &str, at: usize) -> bool {
    text.get(at..)
        .and_then(|suffix| suffix.chars().next())
        .is_some_and(identifier_continuation)
}

fn identifier_continuation(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric() || !character.is_ascii()
}

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
    let bytes = text.as_bytes();
    let registration_name = "OperationRegistration";
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(text, offset) {
            offset += span.get();
            continue;
        }
        if !identifier_continues_before(text, offset)
            && bytes[offset..].starts_with(registration_name.as_bytes())
        {
            let after_name = offset + registration_name.len();
            if !identifier_continues_at(text, after_name) {
                let body = &text[offset..];
                let after = &text[after_name..];
                let constructor = CONSTRUCTOR_TIERS.iter().find(|(call, _)| {
                    after
                        .trim_start()
                        .starts_with(call.trim_start_matches("::"))
                        || after.starts_with(call)
                });
                if let Some((call, tier)) = constructor {
                    if let Some(id) =
                        first_argument(after, call).and_then(|raw| resolve_id(raw, &consts))
                    {
                        let tier = tier
                            .map(|tier| tier.to_string())
                            .or_else(|| nth_argument(after, call, 1).map(tier_variant));
                        found.push((id, tier));
                    }
                } else if after.trim_start().starts_with('{') {
                    let block = &body[..struct_literal_end(body)];
                    if let Some(id) =
                        field_value(block, "id").and_then(|raw| resolve_id(raw, &consts))
                    {
                        found.push((id, field_value(block, "tier").map(tier_variant)));
                    }
                }
                offset = after_name;
                continue;
            }
        }
        offset += 1;
    }
    parse_named_intrinsic_macros(text, &consts, &mut found);
    parse_positional_intrinsic_macros(text, &consts, &mut found);
    found
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
    let mut consts = BTreeMap::new();
    let bytes = text.as_bytes();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(text, offset) {
            offset += span.get();
            continue;
        }
        if !identifier_continues_before(text, offset) && bytes[offset..].starts_with(b"const") {
            let after_const = offset + "const".len();
            if after_const < bytes.len() && !identifier_continues_at(text, after_const) {
                let mut cursor = after_const;
                while cursor < bytes.len() {
                    if let Some(span) = opaque_span(text, cursor) {
                        cursor += span.get();
                        continue;
                    }
                    if bytes[cursor].is_ascii_whitespace() {
                        cursor += 1;
                    } else {
                        break;
                    }
                }
                let name_start = cursor;
                while cursor < bytes.len() && is_word_byte(bytes[cursor]) {
                    cursor += 1;
                }
                let name = text[name_start..cursor].trim();
                while cursor < bytes.len() {
                    if let Some(span) = opaque_span(text, cursor) {
                        cursor += span.get();
                        continue;
                    }
                    if bytes[cursor].is_ascii_whitespace() {
                        cursor += 1;
                    } else {
                        break;
                    }
                }
                if cursor < bytes.len() && bytes[cursor] == b':' {
                    cursor += 1;
                    let type_start = cursor;
                    while cursor < bytes.len() {
                        if let Some(span) = opaque_span(text, cursor) {
                            cursor += span.get();
                            continue;
                        }
                        if bytes[cursor] == b'=' {
                            break;
                        }
                        cursor += 1;
                    }
                    let declared_type = text[type_start..cursor].trim();
                    if cursor < bytes.len()
                        && bytes[cursor] == b'='
                        && is_string_reference_type(declared_type)
                        && !name.is_empty()
                    {
                        cursor += 1;
                        let val_start = cursor;
                        while cursor < bytes.len() {
                            if let Some(span) = opaque_span(text, cursor) {
                                cursor += span.get();
                                continue;
                            }
                            if bytes[cursor] == b';' {
                                break;
                            }
                            cursor += 1;
                        }
                        let val_str = &text[val_start..cursor];
                        if let Some(literal) = string_literal(val_str) {
                            consts.insert(name.to_string(), literal);
                        }
                        offset = cursor + 1;
                        continue;
                    }
                }
            }
        }
        offset += 1;
    }
    consts
}

fn is_string_reference_type(declared_type: &str) -> bool {
    let masked = mask_comments_and_strings(declared_type);
    let compact: String = masked
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    matches!(compact.as_str(), "&str" | "&'staticstr")
}

fn string_literal(text: &str) -> Option<String> {
    crate::source_scan::string_literals(text)
        .into_iter()
        .next()
        .map(str::to_string)
}

/// Read one `field: value,` from a struct literal body.
pub(crate) fn field_value<'a>(block: &'a str, field: &str) -> Option<&'a str> {
    let bytes = block.as_bytes();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if let Some(span) = opaque_span(block, offset) {
            offset += span.get();
            continue;
        }
        if !identifier_continues_before(block, offset)
            && bytes[offset..].starts_with(field.as_bytes())
        {
            let after_field = offset + field.len();
            if !identifier_continues_at(block, after_field) {
                let mut cursor = after_field;
                while cursor < bytes.len() {
                    if let Some(c_span) = is_comment_span(block, cursor) {
                        cursor += c_span;
                        continue;
                    }
                    if bytes[cursor].is_ascii_whitespace() {
                        cursor += 1;
                    } else {
                        break;
                    }
                }
                if cursor < bytes.len() && bytes[cursor] == b':' {
                    cursor += 1;
                    while cursor < bytes.len() {
                        if let Some(c_span) = is_comment_span(block, cursor) {
                            cursor += c_span;
                            continue;
                        }
                        if bytes[cursor].is_ascii_whitespace() {
                            cursor += 1;
                        } else {
                            break;
                        }
                    }
                    let val_start = cursor;
                    let mut depth = 0usize;
                    let mut val_end = cursor;
                    while cursor < bytes.len() {
                        if let Some(span) = opaque_span(block, cursor) {
                            cursor += span.get();
                            val_end = cursor;
                            continue;
                        }
                        let c = bytes[cursor];
                        match c {
                            b'(' | b'[' | b'{' => depth += 1,
                            b')' | b']' | b'}' if depth > 0 => depth -= 1,
                            b')' | b']' | b'}' if depth == 0 => break,
                            b',' if depth == 0 => break,
                            _ => {}
                        }
                        cursor += 1;
                        val_end = cursor;
                    }
                    return Some(block[val_start..val_end].trim());
                }
            }
        }
        offset += 1;
    }
    None
}

pub(crate) fn resolve_id(raw: &str, consts: &BTreeMap<String, String>) -> Option<String> {
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
    fn an_intrinsic_constructor_registration_is_parsed() {
        let parsed = parse_registrations(
            r#"
    inventory::submit! {
        vyre_foundation::operation::OperationRegistration::intrinsic(
            OP_ID,
            crate::hardware::catalog::U32_UNARY_SIGNATURE,
            Some(|| bit_reverse_u32("input", "out", 4)),
            Some(test_inputs),
            Some(expected_output),
        )
    }
    const OP_ID: &str = "vyre-primitives::hardware::bit_reverse_u32";
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "vyre-primitives::hardware::bit_reverse_u32".to_string(),
                Some("Intrinsic".to_string())
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

    #[test]
    fn registration_names_inside_opaque_spans_are_ignored() {
        let parsed = parse_registrations(
            r#"
            const DESCRIPTION: &str =
                "OperationRegistration::library(\"test::string\")";
            // OperationRegistration::library("test::line_comment");
            /* outer OperationRegistration::library("test::block_comment");
               /* nested OperationRegistration::library("test::nested_comment"); */
            */
            OperationRegistration::library("vyre-libs::hash::crc32");
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

    #[test]
    fn string_consts_handles_raw_strings_and_comments() {
        let source = r##"
            /// doc comment with const IGNORED: &str = "bad";
            /* const ALSO_IGNORED: &str = "bad"; */
            const OP1: &str = "op::one";
            const OP2: &'static str = r#"op::two"#;
            const OP3: &str = /* comment with ; */ "op::three;with_semicolon";
        "##;
        let consts = string_consts(source);
        assert_eq!(consts.get("OP1").map(String::as_str), Some("op::one"));
        assert_eq!(consts.get("OP2").map(String::as_str), Some("op::two"));
        assert_eq!(
            consts.get("OP3").map(String::as_str),
            Some("op::three;with_semicolon")
        );
        assert_eq!(consts.get("IGNORED"), None);
        assert_eq!(consts.get("ALSO_IGNORED"), None);
        assert_eq!(
            string_consts("const NOT_AN_ID: &restrict::Name = \"bad\";").get("NOT_AN_ID"),
            None
        );
    }

    /// Byte-oriented scans must cross UTF-8 identifiers without creating an
    /// invalid `str` boundary before a later registration token.
    #[test]
    fn registrations_after_non_ascii_source_are_parsed() {
        let parsed = parse_registrations(
            r#"
fn λ() {}
const OP: &str = "vyre-libs::unicode::library";
OperationRegistration::library(OP);

#[cfg(test)]
mod δοκιμή {
    OperationRegistration::library("test::unicode_fixture");
}

const INTRINSIC: &str = "vyre-primitives::unicode::intrinsic";
submit_hardware_intrinsic! {
    id: INTRINSIC,
    signature: F32_UNARY_SIGNATURE,
}
"#,
        );

        assert_eq!(
            parsed,
            vec![
                (
                    "vyre-libs::unicode::library".to_string(),
                    Some("Library".to_string())
                ),
                (
                    "vyre-primitives::unicode::intrinsic".to_string(),
                    Some("Intrinsic".to_string())
                ),
            ]
        );
    }

    /// Unicode identifier scalars adjacent to canonical token text must not
    /// turn an identifier substring into registration syntax.
    #[test]
    fn registration_tokens_inside_unicode_identifiers_are_ignored() {
        let parsed = parse_registrations(
            r#"
λOperationRegistration::library("bad::constructor");
OperationRegistrationλ::library("bad::constructor_suffix");
λsubmit_hardware_intrinsic! {
    id: "bad::macro",
    signature: F32_UNARY_SIGNATURE,
}
submit_hardware_intrinsicλ! {
    id: "bad::macro_suffix",
    signature: F32_UNARY_SIGNATURE,
}
λconst BAD: &str = "bad::const";
constλ BAD_SUFFIX: &str = "bad::const_suffix";
OperationRegistration::library(BAD_SUFFIX);
OperationRegistration::library(BAD);
OperationRegistration {
    λid: "bad::field",
    tier: OperationTier::Library,
}
OperationRegistration {
    idλ: "bad::field_suffix",
    tier: OperationTier::Library,
}
OperationRegistration::library("good::registration");
"#,
        );

        assert_eq!(
            parsed,
            vec![(
                "good::registration".to_string(),
                Some("Library".to_string())
            )]
        );
        assert!(string_consts("λconst BAD: &str = \"bad::const\";").is_empty());
        assert_eq!(field_value("{ λid: \"bad::field\" }", "id"), None);
    }
}
