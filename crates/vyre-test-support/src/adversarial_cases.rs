//! The case loop the adversarial contract suites share.
//!
//! An adversarial suite is a table of `operands => expected` rows measured
//! against one `cpu_ref` oracle the suite brings into scope. The loop that
//! turns a row into a `#[test]` carries no operation knowledge, so one
//! definition serves every operand arity: the row states the arguments exactly
//! as the oracle takes them, references included.
//!
//! Three near-identical loops existed, one per arity, copied across the suites
//! that used them. They differed only in which positions the loop inserted a
//! `&` into, which is a fact about the oracle and belongs at the row.

/// Turn `name: args... => expected, message;` rows into one `#[test]` each.
///
/// `cpu_ref` is resolved at the call site, so the suite decides which oracle
/// the rows are measured against.
#[macro_export]
macro_rules! adversarial_cpu_ref_cases {
    ($($name:ident: $($arg:expr),+ => $expected:expr, $message:expr;)+) => {
        $(
            #[test]
            fn $name() {
                let expected = $expected;
                let actual = cpu_ref($($arg),+);
                assert_eq!(actual, expected, "{}", $message);
            }
        )+
    };
}
