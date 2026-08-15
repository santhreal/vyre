//! The one coverage ledger for a shared case table.
//!
//! A case table shared by two crates has a failure mode a plain include cannot
//! catch: the table declares a group, one crate grows an arm for it, the other
//! does not, and the crate without the arm still passes because nothing there
//! ever mentions the group. That is the same hole a per-crate copy of the corpus
//! had, moved one level up.
//!
//! [`ArmCoverage`] closes it. An arm records the groups it asserted, the declared
//! set is read from the table at run time rather than listed here, and
//! [`ArmCoverage::assert_covers_declared_table`] names every declared group this
//! crate has no branch for. Adding a group to a table therefore turns red every
//! crate that does not answer for it.
//!
//! Two tables already need this ledger (dense byte-tile matvec in
//! `tests/support/dense_matvec_cases.rs` and the exploded-IFDS reference in
//! [`crate::exploded_ifds_cases`]), so it lives here rather than once per table.
//! The table stays the owner of what its groups are; this type owns only the
//! bookkeeping and the failure messages.

use std::collections::BTreeSet;

/// Which declared case groups one crate's arm actually asserted.
pub struct ArmCoverage {
    table: String,
    owner: String,
    declared: Vec<&'static str>,
    covered: BTreeSet<&'static str>,
    asserted_cases: usize,
    min_cases: usize,
}

impl ArmCoverage {
    /// Start an empty ledger over `declared`, the group names the table returned
    /// on this run.
    ///
    /// `table` names the corpus in failure messages and `owner` is where a
    /// reader edits it. `min_groups` and `min_cases` are floors: a table
    /// enumeration that breaks returns almost nothing, and an arm is trivially
    /// complete over an empty declared set, so a collapsed table has to fail
    /// rather than report a clean sweep.
    ///
    /// # Panics
    /// Panics when fewer than `min_groups` groups are declared, or when one name
    /// is declared twice: coverage is keyed on the name, so a duplicate lets one
    /// arm answer for both.
    #[must_use]
    pub fn new(
        table: &str,
        owner: &str,
        declared: Vec<&'static str>,
        min_groups: usize,
        min_cases: usize,
    ) -> Self {
        assert!(
            declared.len() >= min_groups,
            "Fix: the {table} case table declares only {} group(s); at least {min_groups} are expected, so the enumeration in {owner} is broken and every arm would pass by covering nothing.",
            declared.len()
        );
        let mut unique = BTreeSet::new();
        for name in &declared {
            assert!(
                unique.insert(*name),
                "Fix: {table} case group `{name}` is declared twice in {owner}; coverage is keyed on the name, so a duplicate lets one arm answer for both."
            );
        }
        Self {
            table: table.to_string(),
            owner: owner.to_string(),
            declared,
            covered: BTreeSet::new(),
            asserted_cases: 0,
            min_cases,
        }
    }

    /// Record that this arm asserted all `cases` cases of `group`.
    ///
    /// # Panics
    /// Panics when `cases` is zero, since recording an empty group would claim
    /// coverage for nothing.
    pub fn record(&mut self, group: &'static str, cases: usize) {
        assert!(
            cases > 0,
            "Fix: {} case group `{group}` declares no cases in {}, so recording it as covered asserts nothing.",
            self.table,
            self.owner
        );
        self.covered.insert(group);
        self.asserted_cases += cases;
    }

    /// Fail unless this arm covered every declared group with enough cases.
    ///
    /// # Panics
    /// Panics naming each declared group the arm has no branch for, and when the
    /// arm asserted fewer than the case floor in total.
    pub fn assert_covers_declared_table(&self) {
        let missing: Vec<&str> = self
            .declared
            .iter()
            .copied()
            .filter(|name| !self.covered.contains(name))
            .collect();
        assert!(
            missing.is_empty(),
            "Fix: this crate has no {} arm for declared case group(s) {missing:?}. Either add the arm or delete the group from {}; a declared group with no arm is a case table nobody runs.",
            self.table,
            self.owner
        );
        assert!(
            self.asserted_cases >= self.min_cases,
            "Fix: this crate's {} arms asserted only {} case(s); at least {} are expected.",
            self.table,
            self.asserted_cases,
            self.min_cases
        );
    }
}
