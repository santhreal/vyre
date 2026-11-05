//! WHY: every registered semantic operation records what the executable law
//! vocabulary established about it, and the record is joined to the run rather
//! than accepted as a label.
//!
//! The class this closes is a recorded decision nothing executed. A
//! registration used to satisfy the law gate by carrying any non-placeholder
//! sentence beside its absence, and 272 of them did: 29 distinct sentences
//! covered every operation in the catalog, one string serving 74 registrations.
//! A sentence is not evidence, and neither is a decision a reader can write
//! into agreement with itself.
//!
//! The roster is the live registry, walked at run time, so an operation added
//! to any linked crate is judged here without being named. Every operation is
//! surveyed by executing each family in the provable vocabulary against its own
//! program through the reference oracle, and the recorded decision has to equal
//! the disposition that run derives. There is no allowance table: a new
//! operation whose record disagrees with its run turns this red, and so does an
//! existing one whose program or fixtures change.
//!
//! What this does not catch: a family outside the provable vocabulary, and a
//! law that holds on the fixture cases and fails elsewhere. A confirmation is
//! treated as a necessary condition and never promoted into a declared law for
//! that reason; only a refutation, which is a counterexample, drives a record.

use std::collections::BTreeMap;

use vyre_conform::law_survey::{judge, survey_operation, Disposition};

#[test]
fn every_registered_operation_records_the_disposition_its_run_derives() {
    let registry = vyre_registry_link::operation::live_operation_registry();
    let mut counts: BTreeMap<&'static str, usize> = Disposition::ALL
        .iter()
        .map(|disposition| (disposition.name(), 0usize))
        .collect();
    let mut failures = Vec::new();
    let mut judged = 0usize;

    for entry in registry.iter() {
        judged += 1;
        let survey = survey_operation(&entry);
        *counts
            .get_mut(survey.disposition.name())
            .expect("every disposition name has a counter") += 1;
        for defect in judge(&entry, &survey) {
            failures.push(defect.describe(entry.id));
        }
    }

    assert!(
        judged > 300,
        "the live registry is not linked: {judged} operation(s) reached this binary"
    );
    assert!(
        failures.is_empty(),
        "{} of {judged} operation(s) record a decision their run contradicts:\n{}\ncounts by derived disposition: {counts:?}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(
        counts.get("unrunnable").copied(),
        Some(0),
        "an operation whose law proof cannot run states nothing about itself"
    );
}
