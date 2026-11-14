//! Contracts for the source-derived variant enumeration.
//!
//! Every closure test in this workspace that derives a variant space from
//! source reaches [`top_level_variant_names`]. A short or empty answer from it
//! does not fail loudly on its own: a caller that asserts containment rather
//! than equality passes vacuously, and the class it claims to close stays open.
//! These cases pin the shapes that made it under-report.

#![allow(missing_docs)]

use vyre_test_support::{braced_body, enum_variant_names_in_sources, top_level_variant_names};

/// An attribute belongs to the variant after it.
///
/// This is the case that returned an empty set: an attribute cleared the
/// item-start state and nothing restored it, so a variant carrying
/// `#[error(..)]` was never recorded. `thiserror` enums put one on every
/// variant, so the derived set was empty for all of them.
#[test]
fn a_variant_carrying_an_attribute_is_still_derived() {
    let body = r#"
    /// Doc comment.
    #[error("io_uring {syscall} failed: errno={errno}. Fix: {fix}")]
    IoUringSyscall {
        syscall: &'static str,
        errno: i32,
    },
    /// Second variant.
    #[error("queue at capacity ({depth}). Fix: {fix}")]
    QueueFull {
        depth: u32,
    },
"#;
    let names = top_level_variant_names(body);
    assert_eq!(
        names,
        ["IoUringSyscall", "QueueFull"]
            .into_iter()
            .map(String::from)
            .collect(),
        "Fix: a variant that carries an attribute must still be derived"
    );
}

/// A bracket inside an attribute's string literal is not nesting.
#[test]
fn a_bracket_inside_an_attribute_string_does_not_hide_the_next_variant() {
    let body = r#"
    #[error("index [{index}] is out of range. Fix: {fix}")]
    OutOfRange { index: usize },
    #[error("plain")]
    Plain,
"#;
    assert_eq!(
        top_level_variant_names(body),
        ["OutOfRange", "Plain"]
            .into_iter()
            .map(String::from)
            .collect(),
        "Fix: bracket counting must ignore brackets inside a string literal"
    );
}

/// An escaped quote does not end the string it appears in.
#[test]
fn an_escaped_quote_inside_an_attribute_does_not_end_the_string() {
    let body = r#"
    #[error("saw \"]\" here. Fix: {fix}")]
    Quoted { value: u32 },
    Bare,
"#;
    assert_eq!(
        top_level_variant_names(body),
        ["Quoted", "Bare"].into_iter().map(String::from).collect(),
        "Fix: an escaped quote must not terminate the attribute string"
    );
}

/// Multiple attributes stack on one variant.
#[test]
fn stacked_attributes_do_not_consume_the_variant_they_precede() {
    let body = r#"
    #[non_exhaustive]
    #[error("two attributes")]
    Stacked,
    Next,
"#;
    assert_eq!(
        top_level_variant_names(body),
        ["Stacked", "Next"].into_iter().map(String::from).collect(),
        "Fix: each attribute must be skipped without consuming the variant"
    );
}

/// Payload contents are never mistaken for variants.
#[test]
fn a_payload_type_name_is_not_recorded_as_a_variant() {
    let body = r#"
    #[error("carries a payload")]
    Carrier {
        Inner: u32,
        other: Vec<Nested>,
    },
    Tuple(SomeType, OtherType),
"#;
    assert_eq!(
        top_level_variant_names(body),
        ["Carrier", "Tuple"].into_iter().map(String::from).collect(),
        "Fix: only names declared at variant depth are variants"
    );
}

/// The two functions compose on the shape they are used against in the tree.
#[test]
fn the_body_and_the_variant_scan_agree_on_an_attribute_bearing_enum() {
    let source = r#"
#[derive(Debug)]
#[non_exhaustive]
pub enum Sample {
    /// First.
    #[error("first {a}. Fix: {fix}")]
    First { a: u32 },
    /// Second.
    #[error("second")]
    Second,
}
"#;
    let body =
        braced_body(source, "pub enum Sample {").expect("Fix: the declaration must be found");
    assert_eq!(
        top_level_variant_names(body),
        ["First", "Second"].into_iter().map(String::from).collect(),
        "Fix: a body scan and a variant scan must report the same members"
    );
}

/// X-macro callback tables resolve across files and ignore commas or uppercase
/// words inside string literals.
#[test]
fn xmacro_callback_table_variants_resolve_across_files_and_quoted_commas() {
    let table_file = r#"
macro_rules! for_each_sample_op {
    ($emit:ident) => {
        $emit! {
            Alpha "alpha, FakeVariant" 0x01,
            Beta "beta { Inner }" 0x02,
            Gamma "gamma \"quoted, FakeTwo\"" 0x03,
        }
    };
}
"#;
    let enum_file = r#"
macro_rules! define_sample_op {
    ($($variant:ident $label:literal $tag:literal,)+) => {
        pub enum SampleOp {
            $(
                #[doc = $label]
                $variant,
            )+
        }
    };
}
for_each_sample_op!(define_sample_op);
"#;
    let sources = vec![table_file.to_string(), enum_file.to_string()];
    assert_eq!(
        enum_variant_names_in_sources("SampleOp", &sources),
        ["Alpha", "Beta", "Gamma"]
            .into_iter()
            .map(String::from)
            .collect()
    );
}

/// Every `pub enum` declared in `crates/vyre-spec/src` or `crates/vyre-ir/src`
/// resolves a non-empty variant set through `public_api_variant_names`.
#[test]
fn every_spec_enum_resolves_through_public_api_variant_names() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let mut enum_names = std::collections::BTreeSet::new();
    for candidate_dir in [
        root.join("crates/vyre-ir/src"),
        root.join("crates/vyre-spec/src"),
    ] {
        let mut pending = vec![candidate_dir];
        while let Some(dir) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let source = vyre_test_support::read_source_file_bounded(&path)
                        .expect("Fix: spec source file must be readable");
                    for (pos, _) in source.match_indices("pub enum ") {
                        let line_start = source[..pos].rfind('\n').map_or(0, |idx| idx + 1);
                        if source[line_start..pos].trim_start().starts_with("//") {
                            continue;
                        }
                        let rest = &source[pos + "pub enum ".len()..];
                        let len = rest
                            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                            .unwrap_or(rest.len());
                        if len > 0 && rest[len..].trim_start().starts_with('{') {
                            enum_names.insert(rest[..len].to_string());
                        }
                    }
                }
            }
        }
        if !enum_names.is_empty() {
            break;
        }
    }
    assert!(
        enum_names.len() >= 50,
        "Fix: expected to discover the full spec enum set, found {}",
        enum_names.len()
    );
    for enum_name in &enum_names {
        let variants = vyre_test_support::spec_variant_tables::public_api_variant_names(enum_name);
        assert!(
            !variants.is_empty(),
            "Fix: public_api_variant_names({enum_name:?}) returned empty"
        );
    }
    assert_eq!(
        vyre_test_support::spec_variant_tables::public_api_variant_names("SubgroupReduceOp"),
        ["Add", "And", "Max", "Min", "Mul", "Or", "Xor"]
            .into_iter()
            .map(String::from)
            .collect()
    );
}
