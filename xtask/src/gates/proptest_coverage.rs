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

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Measured floor. 181 tracked files on 2026-08-12.
const FLOOR: usize = 181;

/// Stretch target tracked for the 0.7 release.
const TARGET: usize = 200;

/// How a file declares property tests.
const MARKERS: &[&str] = &[
    "proptest!",
    "use proptest",
    "proptest::",
    "extern crate proptest",
];

pub struct ProptestCoverage;

impl Gate for ProptestCoverage {
    fn name(&self) -> &'static str {
        "proptest-coverage"
    }

    fn help(&self) -> &'static str {
        "tracked source files carrying property tests, against a floor"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
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
                format!("property-test coverage is {} file(s) below the floor of {FLOOR}", FLOOR - carrying),
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
