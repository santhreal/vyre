//! Property-test coverage does not shrink.
//!
//! Property tests are the cheapest way to expose IR, wire-format and optimizer
//! invariants at scale, so the number of files carrying them is a floor rather
//! than a ceiling. This is the one inverted rule in the registry: coverage above
//! the floor is progress and is reported, coverage below it is a deleted test.
//!
//! An earlier version also failed when the count rose, demanding a manual floor
//! bump, so it punished the improvement it exists to encourage and was never
//! wired into CI.
//!
//! The floor is raised deliberately, in a commit that says why. It is never
//! lowered to match a deletion: restore the test instead.

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Measured floor. 174 tracked files, down from 175 on 2026-08-15, itself down
/// from 181 on 2026-08-12. The C frontend and the Rust frontend left the
/// workspace in `1d28c2277f`, and eight of the 670 files that commit deleted
/// carried property tests, measured against its deleted-file list.
///
/// Two more files left the count without losing an invariant.
/// `vyre-reference/tests/dual_reference_property_contracts.rs` generated
/// inputs for the dual evaluator, which `75c0f39de0` deleted along with
/// `dual_impls/**` when the oracle took one canonical evaluator.
/// `9093ab27c8` merged the forward and backward CSR traversal parity files
/// into `vyre-libs-graph/tests/csr_traverse_ir_parity_proptest.rs`, which
/// carries both modules: one file fewer, the same two properties.
///
/// A subject that left the workspace and a consolidation that kept every
/// property are the only two reasons this line may fall. Every other lowering
/// is a deleted test and is refused. The third file `9093ab27c8` dropped,
/// `adversarial_frontier_queue_clear.rs`, had a live subject and was restored
/// as `vyre-libs-graph/tests/proptest_csr_frontier_queue_clear_out.rs`.
const FLOOR: usize = 174;

/// Stretch target tracked for the 0.7 release.
const TARGET: usize = 200;

/// How a file declares property tests.
const MARKERS: &[&str] = &[
    "proptest!",
    "use proptest",
    "proptest::",
    "extern crate proptest",
];

/// The number of property-test files stays at or above the measured floor.
pub struct ProptestCoverage;

impl crate::gate::GateBehavior for ProptestCoverage {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("proptest source files", tree.all_rust().len());
        let mut carrying = 0_usize;
        for file in tree.all_rust() {
            if scan::contains_any(&tree.read(&file)?, MARKERS) {
                carrying += 1;
            }
        }
        report.note(format!(
            "{carrying} file(s) carry property tests (floor {FLOOR}, target {TARGET})"
        ));
        if carrying < FLOOR {
            report.find(Finding::new(
                format!(
                    "property-test coverage is {} file(s) below the floor of {FLOOR}",
                    FLOOR - carrying
                ),
                "restore the deleted property test; lower the floor in \
                 xtask/src/gates/proptest_coverage.rs only with a stated reason for why the \
                 coverage is no longer needed",
            ));
        } else if carrying > FLOOR {
            report.note(format!(
                "{} file(s) above the floor; raise FLOOR to {carrying} to lock the gain",
                carrying - FLOOR
            ));
        }
        Ok(report)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proptest_marker_detection_and_floor_accounting() {
        let sample_source = r#"
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn test_ir_invariants(val in 0..100) {
                    assert!(val >= 0);
                }
            }
        "#;
        assert!(scan::contains_any(sample_source, MARKERS));

        let non_proptest_source = r#"
            #[test]
            fn regular_unit_test() {
                assert_eq!(2 + 2, 4);
            }
        "#;
        assert!(!scan::contains_any(non_proptest_source, MARKERS));

        let carrying = FLOOR - 5;
        let mut report = Report::clean();
        if carrying < FLOOR {
            report.find(Finding::new(
                format!(
                    "property-test coverage is {} file(s) below the floor of {FLOOR}",
                    FLOOR - carrying
                ),
                "restore the deleted property test",
            ));
        }
        assert_eq!(report.findings.len(), 1);
        assert!(report.findings[0]
            .message
            .contains(&format!("5 file(s) below the floor of {FLOOR}")));
    }
}
