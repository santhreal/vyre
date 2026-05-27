//! External proc-macro UI tests for downstream compile behavior.

#[test]
fn proc_macro_ui_contracts() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/pass/skip_builder_passthrough.rs");
    cases.compile_fail("tests/ui/vyre_pass_rejects_non_unit_struct.rs");
    cases.compile_fail("tests/ui/vyre_pass_rejects_duplicate_requires.rs");
    cases.compile_fail("tests/ui/define_op_requires_program.rs");
    cases.compile_fail("tests/ui/algebraic_laws_rejects_union.rs");
}
