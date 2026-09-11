//! The class closed here: a cfg predicate read by whether it mentions `test`
//! rather than by what it compiles into.
//!
//! `#[cfg(not(test))]` names `test` and is the opposite of a test gate: the item
//! it carries exists only in a build without `test`, which is every shipped
//! build. Reading it as test text removed it from the production views, so a
//! panic there counted against nothing and a registration there was invisible
//! to the rules that read the registry. The hygiene scanner already excluded the
//! spelling; the shared span reader did not, so the two disagreed about the same
//! file.

use structure_gate::cfg_test::{cfg_test_items, cfg_test_line_mask, strip_cfg_test_items};

/// One file holding both polarities of the gate plus an ungated item.
const SOURCE: &str = r#"pub fn always_here() {}

#[cfg(not(test))]
pub fn production_only() {
    let ships = 1;
}

#[cfg(all(unix, not(test)))]
pub fn production_on_unix() {}

#[cfg(test)]
mod tests {
    fn double() {}
}
"#;

/// Zero-based index of the one line containing `needle`.
fn line_of(needle: &str) -> usize {
    SOURCE
        .lines()
        .position(|line| line.contains(needle))
        .expect("the fixture must contain the line")
}

/// WHY: against the mention-based reader the two `not(test)` items were stripped
/// with the test module, so a production scan saw neither.
#[test]
fn an_item_compiled_only_without_test_survives_the_strip() {
    let stripped = strip_cfg_test_items(SOURCE);
    for kept in ["always_here", "production_only", "production_on_unix"] {
        assert!(
            stripped.contains(kept),
            "{kept} compiles into a build without test, so a production scan must still read it"
        );
    }
    assert!(
        !stripped.contains("mod tests"),
        "the test module is what the strip is for"
    );
}

/// WHY: the complementary view has to move with the strip, or one caller judges
/// an item as test code while the other judges the same item as production.
#[test]
fn the_test_view_holds_only_the_test_gated_item() {
    let items = cfg_test_items(SOURCE);
    assert!(
        items.contains("mod tests"),
        "the test module belongs to the test view"
    );
    for excluded in ["production_only", "production_on_unix"] {
        assert!(
            !items.contains(excluded),
            "{excluded} is production code and must not read as test text"
        );
    }
}

/// WHY: the line mask is the view that reports a line number, so a rule using it
/// would exempt a production line from its own budget.
#[test]
fn the_line_mask_marks_only_the_test_gated_lines() {
    let mask = cfg_test_line_mask(SOURCE);
    assert!(
        !mask[line_of("let ships")],
        "a line inside a not(test) item is production code"
    );
    assert!(
        mask[line_of("fn double")],
        "a line inside the test module is test code"
    );
}
