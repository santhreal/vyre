//! The class closed here: a malformed macro invocation that compiles, or that
//! fails with a diagnostic the author cannot act on.
//!
//! Every case under `tests/ui` is a rejection the macro must produce, and its
//! `.stderr` file pins the exact text, so changing a diagnostic is a visible
//! test change. The glob is the point: a new case is covered by adding the
//! pair of files, with no list to keep in step and no way to add a case that
//! silently never runs.

#![allow(missing_docs)]

#[test]
fn malformed_macro_inputs_fail_with_actionable_diagnostics() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/*.rs");
}
