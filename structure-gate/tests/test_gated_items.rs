//! Which bytes of a source file are test-gated.
//!
//! WHY: every registration rule in this workspace judges the text that survives
//! `strip_cfg_test_items`, so a byte misclassified here becomes a contract fact.
//! The scanner searched raw text for the attribute, and this crate's own source
//! documents the scanner: a doc comment quoting `#[cfg(test)]` was read as an
//! attribute, the production function under it was deleted from the scanned
//! text, and the aborted search never reached the real `mod tests`. Its
//! registration fixtures were then read as production registrations, which gave
//! two operations a second defining crate and left a third with none.
//!
//! What these do not catch: an attribute this crate cannot parse at all. A
//! malformed one is skipped, and the item under it stays in the production text.

use structure_gate::cfg_test::{cfg_test_items, strip_cfg_test_items};
use structure_gate::registration_text::parse_registrations;

/// A doc comment quoting the attribute is prose.
#[test]
fn an_attribute_quoted_in_a_doc_comment_gates_nothing() {
    let source = "/// Remove every `#[cfg(test)]` item first.\n\
                  pub fn strip() -> usize {\n    7\n}\n\
                  #[cfg(test)]\nmod tests {\n    fn helper() {}\n}\n";

    let stripped = strip_cfg_test_items(source);

    assert!(
        stripped.contains("pub fn strip()"),
        "the documented function is production code: {stripped}"
    );
    assert!(
        !stripped.contains("fn helper()"),
        "the test module is not production code: {stripped}"
    );
    assert!(cfg_test_items(source).contains("fn helper()"));
}

/// An attribute written inside a string literal is data.
#[test]
fn an_attribute_inside_a_string_literal_gates_nothing() {
    let source = "const ATTR: &str = \"#[cfg(test)]\";\n\
                  pub fn produce() -> usize {\n    3\n}\n";

    let stripped = strip_cfg_test_items(source);

    assert!(stripped.contains("pub fn produce()"), "{stripped}");
    assert_eq!(cfg_test_items(source), String::new());
}

/// A registration in a test module is a fixture, and one beside it is not.
///
/// This is the pair that decides where an operation is defined, so both
/// directions are asserted from one file rather than from two.
#[test]
fn a_fixture_registration_is_not_a_production_registration() {
    let source = "/// Registrations look like `#[cfg(test)]` gated code below.\n\
                  const OP_ID: &str = \"libs::real::op\";\n\
                  inventory::submit! {\n    OperationRegistration::library(OP_ID, builder)\n}\n\
                  #[cfg(test)]\nmod tests {\n    inventory::submit! {\n        \
                  OperationRegistration::library(\"libs::ghost::op\", builder)\n    }\n}\n";

    let parsed: Vec<String> = parse_registrations(source)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    assert_eq!(parsed, vec!["libs::real::op".to_string()]);
}

/// An attribute the scanner cannot close does not end the scan.
#[test]
fn an_unclosed_attribute_does_not_hide_a_later_test_module() {
    let source = "#[cfg(\npub fn kept() {}\n\
                  #[cfg(test)]\nmod tests {\n    fn helper() {}\n}\n";

    let stripped = strip_cfg_test_items(source);

    assert!(
        !stripped.contains("fn helper()"),
        "the later test module is still test code: {stripped}"
    );
}

/// A feature named after tests is not the test gate.
#[test]
fn a_feature_whose_name_contains_test_is_production() {
    let source = "#[cfg(feature = \"test-utils\")]\npub fn helper() {}\n";

    let stripped = strip_cfg_test_items(source);

    assert!(stripped.contains("pub fn helper()"), "{stripped}");
}
