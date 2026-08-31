//! The axis, domain, and guards the specialization contracts share.
//!
//! Two suites ask about the same one-axis contract: one proves what a guard set
//! must satisfy, the other compiles a set under it. They disagree only in what
//! they assert, so the guards are declared once. A second copy would have to be
//! kept identical for a precedence case in one suite to mean what a coverage
//! case in the other says.
//!
//! Included with `#[path]` the same way as `tests/support/artifact_fixtures.rs`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use vyre_megakernel::specialization::{
    AxisDomain, GuardTerm, SpecializationAxis, SpecializationContract, VariantGuard,
};

/// The axis every fixture guard reads.
pub(crate) fn tokens() -> SpecializationAxis {
    SpecializationAxis::SymbolicDimension {
        dimension: "tokens".to_string(),
    }
}

/// A contract over [`tokens`] alone, with the stated domain.
pub(crate) fn contract_over(domain: AxisDomain) -> SpecializationContract {
    let mut axes = BTreeMap::new();
    axes.insert(tokens(), domain);
    SpecializationContract::new(axes).expect("Fix: the fixture domain must be declarable.")
}

/// A guard admitting one closed extent range at one precedence.
pub(crate) fn in_range(low: u64, high: u64, precedence: u16) -> VariantGuard {
    VariantGuard::new(
        vec![GuardTerm::InRange {
            axis: tokens(),
            low,
            high,
        }],
        precedence,
    )
}
